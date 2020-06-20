"""
TRIGGERS

trigger.address      = hex2dec('378');
trigger.zero         = 0;
trigger.width        = 0.005; %1 ms trigger signal

%Target responses, left vs right by correct vs error, garbage
trigger.leftcor    = 20;
trigger.rightcor = 21;
trigger.lefterr    = 22;
trigger.righterr   = 23;

trigger.garbageresp     = 29;

%confidence on
trigger.cjon = 32;
%confidece judgment
trigger.cjoff = [11:17]; %11:16 mapping to 1:6, 17 mapping onto garbage

trigger.startblock = 32;
trigger.endblock = 33;

%motion triggers
trigger.rndmotion = 41; 
trigger.cohm = 42;
trigger.irimotion = 43; 

"""
import pickle
import mne
import numpy as np
from collections import namedtuple
from os.path import join
from pymeg.preprocessing import get_meta, preprocess_block, get_epoch, get_events

inpath = "/home/kdesender/meg_data/seqconf/"
savepath = "/home/nwilming/seqconf"

recording = namedtuple("Recording", ["filename", "subject", "session", "block"])
recordings = [
    # warning, pilots have mistake in triggers triggers
    # recording('Pilot01-1_Seqconf_20190123_01.ds', 1, 1, [1,2,3,4]),
    # recording('Pilot01-1_Seqconf_20190123_02.ds', 1, 1, [5,6,7,8]),
    # recording('Pilot01-2_Seqconf_20190124_01.ds', 1, 2, [1,2,3,4]),
    # recording('Pilot01-2_Seqconf_20190124_02.ds', 1, 2, [5,6,7,8]),
    # recording('Pilot02-1_Seqconf_20190123_01.ds', 1, 1, [1,2,3,4]),
    ###recording('Pilot02-1_Seqconf_20190123_02.ds', 1, 1, [5,6,7,8]), #empty recording
    # recording('Pilot02-1_Seqconf_20190123_03.ds', 1, 1, [5,6,7,8]),
    # recording('Pilot02-2_Seqconf_20190124_01.ds', 1, 2, [1,2,3,4]),
    # recording('Pilot02-2_Seqconf_20190124_02.ds', 1, 2, [5,6,7,8]),
    recording("S3-1_Seqconf_20190323_01.ds", 3, 1, [1, 2, 3, 4]),
    recording("S3-1_Seqconf_20190323_02.ds", 3, 1, [5, 6, 7, 8]),
    recording("S3-2_Seqconf_20190324_01.ds", 3, 2, [1, 2, 3, 4]),
    recording("S3-2_Seqconf_20190324_02.ds", 3, 2, [5, 6, 7, 8]),
    recording("S04-1_Seqconf_20190312_01.ds", 4, 1, [1, 2, 3, 4]),
    recording("S04-1_Seqconf_20190312_02.ds", 4, 1, [5, 6, 7, 8]),
    recording("S04-2_Seqconf_20190317_01.ds", 4, 2, [1, 2, 3, 4]),
    recording("S04-2_Seqconf_20190317_02.ds", 4, 2, [5, 6, 7, 8]),
    recording("S05-1_Seqconf_20190316_02.ds", 5, 1, [1, 2]),
    recording("S05-1_Seqconf_20190316_03.ds", 5, 1, [3, 4]),
    recording("S05-1_Seqconf_20190316_04.ds", 5, 1, [5, 6]),
    recording("S05-1_Seqconf_20190316_05.ds", 5, 1, [7, 8]),
    recording("S05-2_Seqconf_20190317_01.ds", 5, 2, [1, 2, 3, 4]),
    recording("S05-2_Seqconf_20190317_02.ds", 5, 2, [5, 6, 7, 8]),
    recording("S06-1_Seqconf_20190316_01.ds", 6, 1, [1, 2, 3, 4]),
    recording("S06-1_Seqconf_20190316_02.ds", 6, 1, [5, 6, 7, 8]),
    recording("S06-2_Seqconf_20190317_01.ds", 6, 2, [1, 2, 3, 4]),
    recording("S06-2_Seqconf_20190317_02.ds", 6, 2, [5, 6, 7, 8]),
    recording("S07-1_Seqconf_20190321_01.ds", 7, 1, [1, 2, 3, 4]),
    recording("S07-1_Seqconf_20190321_02.ds", 7, 1, [5, 6, 7, 8]),
    recording("S07-2_Seqconf_20190326_01.ds", 7, 2, [9, 10, 11]),
    recording("S07-2_Seqconf_20190326_02.ds", 7, 2, [5, 6, 7, 8]),
    recording("S08-1_Seqconf_20190321_01.ds", 8, 1, [9, 10]),
    recording("S08-1_Seqconf_20190321_02.ds", 8, 1, [5, 6, 7, 8]),
    recording("S08-2_Seqconf_20190326_01.ds", 8, 2, [1, 2, 3, 4]),
    recording("S08-2_Seqconf_20190326_02.ds", 8, 2, [5, 6, 7, 8]),
    recording("S09-1_Seqconf_20190323_01.ds", 9, 1, [1, 2, 3, 4]),
    recording("S09-1_Seqconf_20190323_02.ds", 9, 1, [5, 6, 7, 8]),
    recording("S09-2_Seqconf_20190328_01.ds", 9, 2, [1, 2, 3, 4]),
    recording("S09-2_Seqconf_20190328_02.ds", 9, 2, [5, 6, 7, 8]),
    recording("S10-1_Seqconf_20190327_01.ds", 10, 1, [1, 2, 3, 4]),
    recording("S10-1_Seqconf_20190327_02.ds", 10, 1, [5, 6, 7, 8]),
    recording("S10-2_Seqconf_20190328_01.ds", 10, 2, [1, 2, 3, 4]),
    recording("S10-2_Seqconf_20190328_02.ds", 10, 2, [5, 6, 7, 8]),
    recording("S11-1_Seqconf_20190321_01.ds", 11, 1, [1, 2, 3, 4]),
    recording("S11-1_Seqconf_20190321_02.ds", 11, 1, [5, 6, 7, 8]),
    recording("S11-2_Seqconf_20190327_01.ds", 11, 2, [1, 2, 3, 4]),
    recording("S11-2_Seqconf_20190327_02.ds", 11, 2, [5, 6, 7, 8]),
    recording("S12-1_Seqconf_20190323_01.ds", 12, 1, [1, 2, 3, 4]),
    recording("S12-1_Seqconf_20190323_02.ds", 12, 1, [5, 6, 7, 8]),
    recording("S12-2_Seqconf_20190324_01.ds", 12, 2, [1, 2, 3, 4]),
    recording("S12-2_Seqconf_20190324_02.ds", 12, 2, [5, 6, 7, 8]),
    recording("S12-2_Seqconf_20190324_03.ds", 12, 2, [9, 10]),
    recording("S13-1_Seqconf_20190324_01.ds", 13, 1, [1, 2, 3, 4]),
    recording("S13-1_Seqconf_20190324_02.ds", 13, 1, [5, 6, 7, 8]),
    recording("S13-2_Seqconf_20190326_01.ds", 13, 2, [1, 2, 3, 4]),
    recording("S13-2_Seqconf_20190326_02.ds", 13, 2, [5, 6, 7, 8]),
    recording("S14-1_Seqconf_20190323_01.ds", 14, 1, [1, 2, 3, 4]),
    recording("S14-1_Seqconf_20190323_02.ds", 14, 1, [5, 6, 7, 8]),
    recording("S14-2_Seqconf_20190324_01.ds", 14, 2, [1, 2, 3, 4]),
    recording("S14-2_Seqconf_20190324_02.ds", 14, 2, [5, 6, 7, 8]),
    recording("S15-1_Seqconf_20190327_01.ds", 15, 1, [1, 2, 3, 4]),
    recording("S15-1_Seqconf_20190327_02.ds", 15, 1, [5, 6, 7, 8]),
    recording("S15-2_Seqconf_20190328_01.ds", 15, 2, [1, 2, 3, 4]),
    recording("S15-2_Seqconf_20190328_02.ds", 15, 2, [5, 6, 7, 8]),
]


mapping = {
    20: ("response", 1),
    21: ["response", 2],
    22: ["response", 3],
    23: ["response", 4],
    29: ["response", 5],
    32: ["confidence_onset", 0],
    34: ["start_block", 0],
    33: ["end_block", 0],
    11: ["confidence", 1],
    12: ["confidence", 2],
    13: ["confidence", 3],
    14: ["confidence", 4],
    15: ["confidence", 5],
    16: ["confidence", 6],
    41: ["motion_on", 1],
    42: ["coherence_on", 1],
    43: ["irimotion", 1],
}


def submit():
    from pymeg import parallel

    for i in range(len(recordings)):
        parallel.pmap(
            wrap_preprocess,
            [(i,)],
            walltime="15:00:00",
            memory=50,
            nodes=1,
            tasks=4,
            name="PREP_" + recordings[i].filename,
            ssh_to=None,
            env="py36",
        )


def wrap_preprocess(i):
    return preprocess_raw(recordings[i])


def get_hash(subject, session, block, trial):
    """
  2 sessions
  8 blocks
  120 trials per block
  960 per session
  = 1920 trials per subject
  """
    return (
        trial + 120 * (block - 1) + (120 * 8 * (session - 1)) + (1920 * (subject - 1))
    )


def get_blocks(raw):

    meta, timing = get_meta(raw, mapping, {}, 41, 41)
    # meta.loc[:, 'hash'] = np.arange(len(meta))
    # meta.loc[:, 'timing'] = np.arange(len(meta))
    return blocks_from_marker(raw)


def get_preprocessed_block(raw, block):
    start, end = block.start * 60, block.end * 60  # To seconds
    r = raw.copy().crop(start, end)
    print("Processing ", r)
    block_meta, block_timing = get_meta(r, mapping, {}, 41, 41)
    r, ants, artdef = preprocess_block(r)
    return r, ants, artdef


def blocks_from_marker(raw, sfreq=1200.0):
    """
    Find breaks to cut data into pieces.

    Returns a dictionary ob blocks with a namedtuple that
    defines start and end of a block. Start and endpoint
    are inclusive (!).    
    """
    events, _ = get_events(raw)
    onsets = events[events[:, 2] == 34, 0] / sfreq / 60
    ends = events[events[:, 2] == 33, 0] / sfreq / 60

    Block = namedtuple("Block", ["start", "end"])
    blocks = {
        block: Block(start, end) for block, (start, end) in enumerate(zip(onsets, ends))
    }
    return blocks


def filenames(subject, epoch, session, block):
    from os.path import join

    path = join(savepath, "seqconf")
    fname = "S%02i_%s_SESS%i_B%i" % (subject, epoch, session, block)
    return (
        join(path, fname + "-epo.fif.gz"),
        join(path, fname + ".meta"),
        join(path, fname + ".artdef"),
    )


def preprocess_raw(recording):
    raw = mne.io.ctf.read_raw_ctf(join(inpath, recording.filename))
    blocks = get_blocks(raw)
    min_start, max_end = np.min(raw.times), np.max(raw.times)

    for i, block in blocks.items():
        # Cut into blocks

        print("Notch filtering")
        midx = np.where([x.startswith("M") for x in r.ch_names])[0]
        r.load_data()
        r.notch_filter(np.arange(50, 251, 50), picks=midx)

        # Get unique trial indices
        index = get_hash(
            recording.subject,
            recording.session,
            recording.block[i],
            np.arange(1, len(block_meta) + 1),
        )
        block_meta.loc[:, "trial"] = index

        def uniquify(x):
            data = []
            for i in x.values:
                try:
                    data.append(i[0])
                except (IndexError, TypeError) as e:
                    data.append(i)
            return data

        # Sometimes subjects press twice, keep only first event
        block_meta.loc[:, "response"] = uniquify(block_meta.loc[:, "response"])
        block_timing.loc[:, "response_time"] = uniquify(
            block_timing.loc[:, "response_time"]
        )
        block_meta.loc[:, "confidence"] = uniquify(block_meta.loc[:, "confidence"])
        block_timing.loc[:, "confidence_time"] = uniquify(
            block_timing.loc[:, "confidence_time"]
        )
        # Cut into epochs
        for epoch, event, (tmin, tmax), (rmin, rmax) in zip(
            ["stimulus", "response", "confidence"],
            ["coherence_on_time", "response_time", "confidence_time"],
            [(-1, 2.5), (-1.5, 1.5), (-1.5, 1.5)],
            [(-0.5, 1.5), (-0.75, 0.75), (-1, 0.5)],
        ):

            m, s = get_epoch(
                r,
                block_meta,
                block_timing,
                event=event,
                epoch_time=(tmin, tmax),
                reject_time=(rmin, rmax),
                epoch_label="trial",
            )
            artdef["drop_log"] = s.drop_log
            artdef["drop_log_stats"] = s.drop_log_stats()
            if len(s) <= 0:
                continue
            epofname, mfname, afname = filenames(
                recording.subject, epoch, recording.session, recording.block[i]
            )
            s = s.resample(600, npad="auto")
            s.save(epofname)
            m.to_hdf(mfname, "meta")
            pickle.dump(artdef, open(afname, "wb"), protocol=2)
