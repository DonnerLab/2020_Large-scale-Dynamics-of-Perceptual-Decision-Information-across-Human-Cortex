'''
Determine mapping of blocks in data files to blocks in experiment.
'''
from conf_analysis.meg import artifacts, preprocessing
from conf_analysis.behavior import empirical, metadata, keymap
import mne, locale
import numpy as np
locale.setlocale(locale.LC_ALL, "en_US")
from distributed import Executor, as_completed
from distributed import diagnostics
import pickle


executor = Executor("172.18.101.120:8786")


def do_one(filename):
    from conf_analysis.meg import artifacts, preprocessing
    from conf_analysis.behavior import empirical, metadata, keymap
    import mne, locale
    import numpy as np
    import pickle
    locale.setlocale(locale.LC_ALL, "en_US")
    result = {}
    raw = mne.io.read_raw_ctf(filename, system_clock='ignore')
    trials = preprocessing.blocks(raw, full_file_cache=True)
    trl, bl = trials['trial'], trials['block']
    bcnt = 0
    for b in np.unique(bl):
        if len(trl[bl==b]) >= 75:
            result[b] = bcnt
            bcnt +=1
        print((b, bcnt))
    return result

block_map = {}
for snum in range(1, 16):
    filenames = [metadata.get_raw_filename(snum, b) for b in range(4)]
    block_map[snum] = {}
    for session, filename in enumerate(filenames):
        block_map[snum][session] = executor.submit(do_one, filename)

diagnostics.progress(block_map)
block_map = executor.gather(block_map)

pickle.dump(block_map, open('blockmap.pickle', 'w'))
