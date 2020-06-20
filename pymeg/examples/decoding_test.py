'''
Pairwise decoding for the orientation vs. location decoding test.

This script will analyze data from an experiment carried out by Peter and Niklas.
In the experiment three different stimulus types were presented:

    - Small checkerboards at random locations around a semi-circle in the lower
      visual field. The polar angle of each checkerboard uniquely describes each
      stimulus' location and is called `target`.
    - Small gabor patches on the same semi-circle. Again at random locations.
    - A very large central gabor stimulus which varied in orientation. For this
      stimulus the orientation is called `target`.

The purpose of this test was to determine which stimulus setup is better suited
to track sensory processing of the stimulus. The task was for all practical
purposes to stare at the fixation cross.

This script analyses MEG responses to these stimuli by running a decoding analysis
in which trials are assigned either to class A or class B based on their target
value (so orientation or location). Each class was centered on a particular
mean target value and contained trials that were within +-15 degrees of the mean.
We then used a SVM machine with standard scaling and a PCA for dimensionality
reduction to train a classifier in a cross-validated fashion.

The function `pairwise_decoding` does all the heavy lifting.

Functions for loading and pre-processing:
    - get_data
    - get_epochs
    - get_meta
    - align

Functions `execute` and `list_tasks` allow easy parallelization with the
`to_cluster` script.
'''

from os import path
from scipy.io import loadmat
import pandas as pd
import numpy as np
import mne
from pymeg import preprocessing, artifacts
from conf_analysis.meg import decoding
from joblib import Memory
from os.path import expanduser

home = expanduser("~")
memory = Memory(cachedir=path.join(home, 'cache_pymeg'), verbose=0)

'''
First set up some general info to load the data.
'''

paths = {'c':'/home/pmurphy/Decoding_tests/Checker_location/Data/DC1/S1/Behaviour/',
         'go':'/home/pmurphy/Decoding_tests/Gabor_orientation/Data/DC1/S1/Behaviour/',
         'gl':'/home/pmurphy/Decoding_tests/Gabor_location/Data/DC1/S1/Behaviour/'}


# These functions return the mat file for a specific block.
c = lambda block: ('C', path.join(paths['c'], 'DC1_1_%i.mat'%block))
go = lambda block: ('GO', path.join(paths['go'], 'DC1_1_%i.mat'%block))
gl = lambda block: ('GL', path.join(paths['gl'], 'DC1_1_%i.mat'%block))

# Defines order of blocks as they occured in the experiment.
block_mapping = [c(1),  go(1), gl(1),
                 go(2), gl(2), c(2),
                 go(3), c(3),  gl(3)]
block_mapping_2 = [c(4), go(4), gl(4)]



def get_data():
    '''
    Returns a single MNE epochs object and associated meta data for both
    data raw data files.
    '''
    ea = get_epochs('/home/pmurphy/Decoding_tests/meg_data/DC1_TimeScale_20170201_01.ds')
    ea2 = get_epochs('/home/pmurphy/Decoding_tests/meg_data/DC1_TimeScale_20170201_02.ds')
    # Treat both files as if head was in the same position.
    ea2.info['dev_head_t'] = ea.info['dev_head_t']
    # Align with metadata
    ea, df = align(ea, epoch_offset=10)
    ea2, df2 = align(ea2, epoch_offset=0, block_mapping=block_mapping_2)
    df2 = df2.set_index(2025+np.arange(len(df2)))
    meta = pd.concat([df, df2])
    epochs = mne.concatenate_epochs([ea, ea2])
    return epochs, meta


@memory.cache
def get_epochs(filename):
    '''
    Produce epoch object from raw file. E.g. detect eye-tracking artifacts and
    chunk into epochs. This method will cache and save it's results to disk.
    '''
    raw = mne.io.read_raw_ctf(filename)
    # Detect blinks.
    raw.annotations = artifacts.annotate_blinks(raw,
        ch_mapping={'x':'UADC001-3705', # These are the eye-tracking channels in the MEG data.
                    'y':'UADC003-3705',
                    'p':'UADC004-3705'})
    # Finds triggers in default trigger channels
    events = preprocessing.get_events(raw)[0]
    # We are only interested in triggers 11, 31 and 51
    ide =  np.in1d(events[:, 2], [11, 31, 51])
    target_events = events[ide, :]
    # Cut out epochs. Reject peak-to-peak distances with 4e-12 threshold
    epochs = mne.Epochs(raw, target_events, tmin=-0.2, tmax=0.9, baseline=(-0.1, 0), reject=dict(mag=4e-12))
    # Actually load the data and resample to 300Hz.
    epochs.load_data()
    epochs.resample(300, n_jobs=4)
    return epochs


def get_meta(block_mapping=block_mapping):
    '''
    Load meta data from mat files.
    '''
    df = []
    index_offset = 0
    for i, (block, filename) in enumerate(block_mapping):
        data = loadmat(filename)['Behav'][:, 0]
        d = pd.DataFrame({'block':i, 'target':data, 'type':block})
        d = d.set_index(np.arange(250)+index_offset)
        if block=='GO':
            print 'Go'
            d.target-=90
        df.append(d)
        index_offset+=250
    return pd.concat(df)


def align(epochs, epoch_offset=0, block_mapping=block_mapping):
    '''
    Return meta data for an epochs object and align them.
    '''
    df = get_meta(block_mapping)
    df = df.dropna()
    df = df.set_index(np.arange(len(df)))
    dropped = np.where(epochs.drop_log)[0]-epoch_offset
    assert (len(df) == len(epochs) + len(dropped) - epoch_offset)
    dropped = dropped[dropped>=0]
    if epoch_offset > 0:
        epochs.drop(np.arange(epoch_offset))
    df = df.drop(dropped)
    return epochs, df


def get_subset(epochs, df, labels):
    '''
    Select a set of epochs based on labels in meta data.
    '''
    index = np.in1d(df.index.values, labels)
    return epochs[index], df.loc[labels, :]


def pairwise_decoding(epochs, meta, delta_target=15):
    '''
    Decode pairs of average orientations. Do for all possible pairs.
    '''
    # Get all possible target orientations and leave delta_target at each end
    # of the semi-circle
    targets = np.unique(meta.target)
    targets = targets[targets>=(min(targets)+delta_target)]
    targets = targets[targets<=(max(targets)-delta_target)]
    results = []

    for it, target in enumerate(targets):
        # target is the center orientation for class A
        predict_targets = targets[targets>=target][::-1]
        for jt, target2 in enumerate(predict_targets):
            # target2 is the center orientation for class B
            # the next few lines select trials for class A (target) and class B
            # (target2)
            index  = (
                     (
                        ((target-delta_target) < meta.target.values) &
                         (meta.target.values < (target+delta_target))
                     )
                        |
                     (
                        ((target2-delta_target) < meta.target.values) &
                         (meta.target.values < (target2+delta_target))
                     )
                )
            lselect = meta.loc[index].index.values
            ego, edf =  get_subset(epochs, meta, lselect)
            times = ego.times
            threshold = np.mean([target, target2])
            # Class labels:
            labels = (edf.target > threshold).values
            data = ego._data
            # Select time between 80 and 220ms
            idx = (0.08<times) & (times<0.22)
            intm = []
            for id_time, time in zip(np.where(idx)[0], times[idx]):
                # Do Decoding for each time point
                acc = decoding.decode(decoding.clf, data, labels, id_time, [id_time],
                        cv=decoding.cv, collapse=np.mean,
                        relabel_times=times)
                acc['t1'] = target
                acc['t2'] = target2
                intm.append(acc)
            results.append(pd.concat(intm))
    return results


def execute(x):
    '''
    This method executes pairwise decoding for one of the three
    block types. The input argument x specifies the block. Can be
    any of 'C', 'GO', 'GL'.

    This method is part of the interface for the to_cluster script. Calling
    `to_cluster --array decoding_test.py` will execute this method on all
    return values of the `list_task` function below.
    '''
    mne.set_log_level('warning')
    epochs, meta = get_data()
    index = (meta.type==x).values
    ec = epochs[index]
    el = ec.pick_channels(decoding.sensors['occipital'](ec.ch_names))
    mc = meta.loc[index]
    acc = pairwise_decoding(el, mc)
    acc = pd.concat(acc)
    acc.to_hdf('decoding_pairwise_%s.hdf'%x, 'decoding')


def list_tasks(older_than='now'):
    '''
    List tasks to be done in parallel on the cluster.
    '''
    for t in ['C', 'GL', 'GO']:
        yield t
