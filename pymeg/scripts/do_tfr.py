import mne, locale
import numpy as np
import json
import cPickle
import os
import datetime

from meg import tfr

def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)

locale.setlocale(locale.LC_ALL, "en_US")

outstr = 'tfr.hdf'
params = tfr.params_from_json('all_tfr150_parameters.json')
tfr.describe_taper(**params)


def list_tasks(older_than='now'):
    import glob
    filenames = glob.glob('/home/gortega/meg_analysis/preprocess_data/resp*-epo.fif.gz') 
    if older_than == 'now':
        older_than = datetime.datetime.today()
    else:
        older_than = datetime.datetime.strptime(older_than, '%Y%m%d')

    for filename in filenames:
        mod_date = modification_date(filename)
        outname = filename.replace('-epo.fif.gz', outstr)
        try:
            mod_date = modification_date(filename)
            if mod_date>older_than:
                continue
        except OSError:
            pass
        yield filename

def execute(filename):
    print 'Starting TFR for ', filename
    print params
    tfr.tfr(filename, outstr, **params)
    print 'Done with TFR for ', filename
