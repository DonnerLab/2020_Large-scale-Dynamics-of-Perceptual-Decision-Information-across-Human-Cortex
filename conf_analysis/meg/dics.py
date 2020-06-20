
import numpy as np
import mne
from conf_analysis.meg import source_recon as sr
from conf_analysis.behavior import metadata
from pymeg import preprocessing as pymegprepr, tfr

from copy import deepcopy

from scipy import linalg

from mne.beamformer._lcmv import (_prepare_beamformer_input, _setup_picks,
                                  _reg_pinv)
from mne.time_frequency import csd_epochs as mne_csd_epochs
from mne.externals import six
from joblib import Memory
from joblib import Parallel, delayed

import time as clock
from glob import glob
import pandas as pd

from sklearn import linear_model

memory = Memory(cachedir=metadata.cachedir)


@memory.cache
def get_tfr(subject, n_blocks=None):
    from mne.time_frequency.tfr import read_tfrs
    from mne.time_frequency import EpochsTFR
    files = glob('/home/nwilming/conf_meg/S%i/SUB%i*stimulus*-tfr.h5' %
                 (subject, subject))
    if n_blocks is not None:
        files = files[:n_blocks]
    etfr = [read_tfrs(files.pop())[0]]
    for fname in files:
        etfr.append(read_tfrs(fname)[0])
    data = np.concatenate([e.data for e in etfr], 0)
    return EpochsTFR(etfr[0].info, data, etfr[0].times, etfr[0].freqs)


def get_metas_for_tfr(subject):
    files = glob('/home/nwilming/conf_meg/S%i/SUB%i*stimulustfr.hdf5' %
                 (subject, subject))
    metas = []
    for f in files:
        sub, sess, block, _ = f.split('/')[-1].split('_')
        sess = int(sess[1:])
        block = int(block[1:])
        metas.append(get_meta_for_block(subject, sess, block))
    return pd.concat(metas)


def get_meta_for_block(subject, session, block):
    meta = metadata.get_epoch_filename(
        subject, session, block, 'stimulus', 'meta')
    meta = pymegprepr.load_meta([meta])
    return meta[0]


def get_tfr_filename_for_trial(meta, trial):
    from itertools import product
    meta = meta.loc[trial]
    blocks = meta.block_num.values
    session = meta.session_num.values
    subject = meta.snum.values

    combinations = set(zip(subject, session, blocks))
    return [('/home/nwilming/conf_meg/S%i/SUB%i_S%i_B%i_stimulustfr.hdf5' %
             (sub, sub, sess - 1, block),
             meta.query('snum==%i & session_num==%i & block_num==%i' %
                        (sub, sess, block)))
            for sub, sess, block in combinations]


def get_tfr_array(meta, freq=(0, 100), channel=None, tmin=None, tmax=None,
                  baseline=None):
    '''
    Load many saved tfrs and return as a numpy array.

    Inputs
    ------
        filenames: List of TFR filenames
        freq:  tuple that specifies which frequencies to pull from TFR file.
        channel: List of channels to include
        tmin & tmax: which time points to include
        baseline: If func it will be applied to each TFR file that is being loaded.
    '''
    filenames = get_tfr_filename_for_trial(meta, meta.index.values)
    fname, block = filenames[0]
    out = tfr.read_chunked_hdf(fname, freq=freq, channel=channel,
                               tmin=tmin, tmax=tmax, epochs=block.index.values)
    freqs, times, events = out['freqs'], out['times'], out['events']
    dfs = [out['data']]
    read_epochs = [events]
    for fname, block in filenames[1:]:
        try:
            out = tfr.read_chunked_hdf(fname, freq=freq, channel=channel,
                                       tmin=tmin, tmax=tmax,
                                       epochs=block.index.values)
            f, t, e = out['freqs'], out['times'], out['events']
            assert(all(freqs == f))
            assert(all(times == t))
            read_epochs.append(e)
            dfs.append(out['data'])
        except RuntimeError as e:
            print(e)
            assert e.msg.contains('None of the requested epochs are')
    return freqs, times, np.concatenate(read_epochs), np.concatenate(dfs, 0)


@memory.cache
def make_csds(epochs,  f, times, t_noise, f_smooth, subject):
    '''
    Compute Beamformer filters for time-points in TFR.
    '''
    fmin, fmax = f - f_smooth, f + f_smooth
    print("Computing noise csd")
    noise_csd = get_noise_csd(epochs, fmin, fmax, t_noise)

    print("Computing data csd with")
    data_csd = mne_csd_epochs(epochs, 'multitaper',
                              fmin=fmin,
                              fmax=fmax,
                              fsum=True,
                              tmin=times[0],
                              tmax=times[1])
    return f, noise_csd, data_csd


@memory.cache
def get_noise_csd(epochs, fmin, fmax, times):
    return mne_csd_epochs(epochs, 'multitaper', fmin, fmax,
                          fsum=True, tmin=times[0],
                          tmax=times[1])


def apply_dics_filter(data_csd, F, meta, filename, subject, n_jobs=1):
    '''
    Apply beamformer to TFR data

    Filters is a dictionary of CSDs, where keys are timepoints.
    f is the frequency of interest
    meta is a meta structure giving information about trials
    filename indicates which memmap file to use for storing source info

    '''
    meta = meta.copy()
    num_epochs = meta.shape[0]
    forward, bem, source, trans = sr.get_leadfield(subject)
    n_source = np.sum([s['nuse'] for s in source])
    sp_shape = output_shape(subject, meta, (-0.75, 1.4), (F, F))

    # memmap results
    source_pow = np.memmap(filename, dtype='float32', mode='w+',
                           shape=sp_shape)
    A = dics_filter(forward, data_csd)
    del forward, bem, source, trans

    print('Applying in parallel')

    meta.loc[:, 'linear_index'] = np.arange(num_epochs)

    def get_jobs():
        for (b, s), m in meta.groupby(['session_num', 'block_num']):
            assert(np.unique(np.diff(m.linear_index.values)) == [1])
            epoch_offset = m.loc[:, 'linear_index'].min()
            args = (source_pow, m, (F, F),
                    (-0.75, 1.4), A, epoch_offset, 0)
            yield delayed(apply_one_filter)(*args)

    epoch_order = Parallel(n_jobs=n_jobs)(get_jobs())
    return source_pow, epoch_order


def apply_one_filter(source_pow, meta, freq, time, A, offset, t_offset):
    '''
    Apply a CSD to some time frequency data that is loaded within the function.

    source_pow: Array (epchs, num_sources, time)
        Array into which to save source-recon (can be memmapped)
    meta: metadata that defines which epochs to load from tfr
    freq: (F_low, F_high) which frequencies to load (will be averaged over)
    time: (t_low, t_high) which time_points to load
    A: beam forming filter
    offset: int
        This function will load TFR data for different epochs and time points.
        This offset determines where in source_pow epochs will be stored.
        >>> source_pow[offset+epoch, :, time] = srdata
        Where epoch is a linear index throug meta (e.g. not the unique trial identifiers)
    t_offset: int
        Same as offset but for time points:
        >>> source_pow[epoch, :, toffset+time] = srdata 
    '''

    # Load source data
    freqs, times, epochs, tfrdata = get_tfr_array(meta, freq=freq,
                                                  tmin=time[0],
                                                  tmax=time[1])

    # tfrdata is num_epochs x channels x F x time
    tfrdata = tfrdata.mean(2)  # now num_epochs x channels x time
    n_time = tfrdata.shape[-1]
    epochs_order = []
    indices = []

    for i, (epoch, Xsensor) in enumerate(zip(epochs, tfrdata), offset):
        # Iterate over epochs, Xsensor = 269 channels x time
        Xsource = np.dot(A, Xsensor)  # 8k = (8k x 269) * (269, time)
        source_pow[i, :, t_offset:t_offset +
                   n_time] = Xsource * np.conj(Xsource)
        epochs_order.append(epoch)
        indices.append(i)
    return indices, epochs_order, list(times)


def apply_tfr(A, source_pow, tfrdata, epochs):
    indices, epochs_order = [], []
    for i, (epoch, Xsensor) in enumerate(zip(epochs, tfrdata)):
        # Iterate over epochs, Xsensor = 269 channels x time
        Xsource = np.dot(A, Xsensor)  # 8k = (8k x 269) * (269, time)
        source_pow[i, :, :] = Xsource * np.conj(Xsource)
        epochs_order.append(epoch)
        indices.append(i)
    return np.array(indices), np.array(epochs_order)


def output_shape(subject, meta, time, freq):
    '''
    Return the shape of the memmap array that is used for saving the single
    trial power estimates.
    Shape is: Number of sources, number of time points, number of epochs
    '''
    freqs, times, epochs, tfrdata = get_tfr_array(meta.iloc[0:1, :],
                                                  freq=freq,
                                                  tmin=time[0],
                                                  tmax=time[1])
    n_epochs = len(meta)
    n_time = tfrdata.shape[-1]
    forward, bem, source, trans = sr.get_leadfield(subject)
    n_source = np.sum([s['nuse'] for s in source])
    return n_epochs, n_source, n_time


def stc_from_memmap(data, subject):
    forward, bem, source, trans = sr.get_leadfield(subject)
    verts = [source[0]['vertno'], source[1]['vertno']]
    tmin = -0.75
    tstep = 0.01666667
    stc = mne.SourceEstimate(data, verts, tmin=tmin,
                             tstep=tstep, subject='S%02i' % subject)
    return stc


def extract_label(data, source, label):
    verts = np.concatenate([source[0]['vertno'], source[1]['vertno']])
    assert(len(verts) == data.shape[1])
    idx = np.in1d(verts, label.vertices)
    if sum(idx) == 0:
        raise RuntimeError('Label not in source space')
    
    return verts[idx], data[:, idx, :]


def get_label_dataframe(meta, data, index, times, source, labels,
                        baseline=None, norm='mean'):
    '''
    Extract a set of labels from memmapped sources and align
    with meta

    meta: DataFrame
        Trial metadata that is indexed by unique trial hash
    data: array, (#epochs x #sources x #times)
        Source reconstructed epochs
    index: dict
        maps unique trial hash to index in data
    times: list
        list of time points that map to last dim in data
    source: mne source space
        Source space used for the source reconstruction
    labels: list of mne labels
    baseline: func
        A function that performs baseline correction. Will be called with:
            >>> func(meta, data, epochs, times)
    '''
    times = np.asarray(times)
    frames = []
    trials = meta.index.values
    idx_array = [index[trial_hash] for trial_hash in meta.index.values]
    for label in labels:
        try:
            vertices, d_label = extract_label(data, source, label)
            print(d_label.shape)
            d_label = d_label[idx_array, :, :]
            if baseline is not None:
                d_label = baseline(meta, d_label, meta.index.values, times)
            # Find voxel with max change
            if norm == 'max':
                idx = np.argmax(np.abs(d_label))
                e, vx, t = np.unravel_index(idx, d_label.shape)
                d_label = d_label[:, vx, :]
                df = pd.DataFrame(d_label, index=meta.index.values,
                                  columns=times).stack()
                #print df.head()
                df.name = label.name
                frames.append(df)
            elif norm == 'none':
                for i, vertex in enumerate(vertices):
                    df = pd.DataFrame(d_label[:, i, :],
                                      index=meta.index.values,
                                      columns=times).stack().reset_index()
                    df.columns = ['trial', 'time', label.name]                    
                    df.loc[:, 'vertex'] = vertex
                    df = df.set_index(['trial', 'time', 'vertex'])
                    frames.append(df)
            else:
                d_label = d_label.mean(1)
                df = pd.DataFrame(d_label, index=meta.index.values,
                                  columns=times).stack()
                df.name = label.name
                frames.append(df)
        except RuntimeError as e:
            pass
    frames = pd.concat(frames, axis=1)
    frames.columns = [f.replace('lh.wang2015atlas.', '')
                       .replace('rh.wang2015atlas.', '')
                      for f in frames.columns]
    return frames


def selector(data, meta, epochs, times):
    '''
    Predict contrast with a bunch from voxels
    '''
    cvals = np.stack(meta.loc[epochs, 'contrast_probe'])
    reg = linear_model.LinearRegression()
    out = np.nan * np.ones((data.shape[0], data.shape[-1]))
    for sample in range(10):
        for ti, time_point in enumerate(times):
            y = cvals[:, sample]
            X = data[:, :, ti]
            fit = reg.fit(X, y)
            out[:, ti] = np.dot(X, fit.coef_)
    return out


def baseline(meta, data, epochs, times):
    idx = (-0.5 < times) & (times < -0.1)
    base = data[:, :, idx].mean(-1).mean(0)[np.newaxis, :, np.newaxis]
    return (data - base) / base


@memory.cache
def dics_filter(forward, data_csd, reg=0.05):
    '''
    forward contains the lead field
    data_csd contains the CSD of interest

    Assume free orientation lead field.
    '''
    Cm = data_csd.data.copy()
    #Cm = Cm.real
    #Cm_inv, _ = _reg_pinv(Cm, reg)
    Cm_inv = np.linalg.pinv(Cm + reg * np.eye(Cm.shape[0]))

    source_loc = forward['source_rr']
    source_ori = forward['source_nn']
    As = np.nan * np.ones((source_loc.shape[0], Cm.shape[0]))
    for i, k in enumerate(range(0, source_ori.shape[0], 3)):
        L = forward['sol']['data'][:, k:k + 3]
        A = np.dot(
            np.dot(
                linalg.pinv(
                    np.dot(
                        np.dot(L.T, Cm_inv),
                        L)),
                L.T), Cm_inv)

        # print A.min(), A.max()
        Aw = np.dot(np.dot(A, Cm), np.conj(A).T)
        v, h = np.linalg.eig(Aw)
        # print h, v
        A = np.dot(A.T, h[:, 0])
        As[i, :] = A
    return As
