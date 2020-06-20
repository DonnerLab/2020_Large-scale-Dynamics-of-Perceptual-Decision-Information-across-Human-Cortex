'''
Preprocess an MEG data set.

The idea for preprocessing MEG data is modelled around a few aspects of the
confidence data set:
    1. Each MEG dataset is accompanied by a DataFrame that contains metadata for
       each trial.
    2. Trial meta data can be matched to the MEG data by appropriate triggers.
    3. Each MEG datafile contains several blocks of data that can be processed
       independently.

This leads to the following design:
    1. MEG is cut into recording blocks and artifact detection is carried out
    2. Each processed block is matched to meta data and timing (event on- and offsets)
       data is extracted from the MEG and aligned with the behavioral data.
    3. Data is epoched. Sync with meta data is guaranteed by a unique key for
       each trial that is stored along with the epoched data.
'''
import logging
import mne
import numpy as np
import os
import pandas as pd
import pickle

from conf_analysis.behavior import empirical
from conf_analysis.behavior import metadata
from itertools import product
from joblib import Memory
from mne.transforms import apply_trans

from pymeg import preprocessing as pymegprepr
from functools import reduce

memory = Memory(location=metadata.cachedir)
import locale
locale.setlocale(locale.LC_ALL, "en_US")


def one_block(snum, session, block_in_raw, block_in_experiment):
    '''
    Preprocess a single block and save results.

    Parameters
    ----------
        snum, session : int
    Subject number and session number
        raw : mne.io.Raw object
    Raw data for an entire session of a subject.
        block_in_raw, block_in_experiment : int

    Each succesfull session consists out of five blocks, yet a sessions MEG
    data file sometimes contains more. This happens, for example, when a block
    is restarted. 'block_in_raw' refers to the actual block in a raw file, and
    block_in_experiment to the block in the metadata that block_in_raw should
    be mapped to. block_in_experiment will be used for saving.
    '''

    try:

        art_fname = metadata.get_epoch_filename(
            snum, session,
            block_in_experiment,
            None, 'artifacts')

        data = empirical.load_data()
        data = empirical.data_cleanup(data)

        filename = metadata.get_raw_filename(snum, session)
        raw = mne.io.read_raw_ctf(filename, system_clock='ignore')
        trials = blocks(raw)
        if not (block_in_raw in np.unique(trials['block'])):
            err_msg = 'Error when processing %i, %i, %i, %i, data file = %s' % (
                snum, session, block_in_raw, block_in_experiment, filename)
            raise RuntimeError(err_msg)

        # Load data and preprocess it.
        logging.info('Loading block of data: %s; block: %i' %
                     (filename, block_in_experiment))
        r, r_id = load_block(raw, trials, block_in_raw)
        r_id['filname'] = filename
        print(('Working on:', filename, block_in_experiment, block_in_raw))
        logging.info('Starting artifact detection')

        r, ants, artdefs = pymegprepr.preprocess_block(r)
        #r.annotations = r
        print('Notch filtering')
        midx = np.where([x.startswith('M') for x in r.ch_names])[0]
        r.notch_filter(np.arange(50, 251, 50), picks=midx)
        logging.info('Aligning meta data')
        meta, timing = get_meta(data, r, snum, block_in_experiment, filename)
        idx = np.isnan(meta.response.values)
        meta = meta.loc[~idx, :]
        timing = timing.loc[~idx, :]
        artdefs['id'] = r_id
        filenames = []
        for epoch, event, (tmin, tmax), (rmin, rmax) in zip(
                ['stimulus', 'response', 'feedback'],
                ['stim_onset_t', 'button_t',
                 'meg_feedback_t'],
                [(-.75, 1.5), (-1.5, 1), (-1, 1)],
                [(0, 1), (-1, 0.5), (-0.5, 0.5)]):

            logging.info('Processing epoch: %s' % epoch)
            try:
                m, s = pymegprepr.get_epoch(r, meta, timing,
                                            event=event, epoch_time=(
                                                tmin, tmax),
                                            reject_time=(rmin, rmax),
                                            )
            except RuntimeError as e:
                print(e)
                continue

            if len(s) > 0:
                epo_fname = metadata.get_epoch_filename(snum, session,
                                                        block_in_experiment, epoch, 'fif')
                epo_metaname = metadata.get_epoch_filename(snum, session,
                                                           block_in_experiment, epoch, 'meta')
                s = s.resample(600, npad='auto')
                s.save(epo_fname)
                m.to_hdf(epo_metaname, 'meta')
                r_id[epoch] = len(s)
                filenames.append(epo_fname)
        pickle.dump(artdefs, open(art_fname, 'wb'), protocol=2)

    except MemoryError:
        print((snum, session, block_in_raw, block_in_experiment))
        raise RuntimeError('MemoryError caught in one block ' + str(snum) + ' ' + str(
            session) + ' ' + str(block_in_raw) + ' ' + str(block_in_experiment))
    return 'Finished', snum, session, block_in_experiment, filenames


def get_meta_from_ds():
    import pickle
    block_map = pickle.load(
        open('/home/nwilming/conf_analysis/required/blockmap.pickle', 'rb'))
    metas = []
    for subject in range(1, 15):
        for session in range(4):
            for block_in_raw, block_in_experiment in block_map[subject][session].items():
                m = get_block_meta(subject, session, block_in_raw, block_in_experiment)
                metas.append(m)
                m.to_hdf('/home/nwilming/S%i_SESS%i_B%i_timing.hdf'%(subject, session, block_in_experiment), 'df')
    return metas


def get_block_meta(snum, session, block_in_raw, block_in_experiment):
    data = empirical.load_data()
    data = empirical.data_cleanup(data)
    filename = metadata.get_raw_filename(snum, session)
    raw = mne.io.read_raw_ctf(filename, system_clock='ignore')
    trials = blocks(raw)
    if not (block_in_raw in np.unique(trials['block'])):
        err_msg = 'Error when processing %i, %i, %i, %i, data file = %s' % (
            snum, session, block_in_raw, block_in_experiment, filename)
        raise RuntimeError(err_msg)
    r, _ = load_block(raw, trials, block_in_raw)
    meta, timing = get_meta(data, r, snum, block_in_experiment, filename)
    return pd.concat((meta, timing), axis=1)


def blocks(raw, full_file_cache=False):
    '''
    Return a dictionary that encodes information about trials in raw.
    '''
    if full_file_cache:
        trigs, buts = pymegprepr.get_events_from_file(raw.info['filename'])
    else:
        trigs, buts = pymegprepr.get_events(raw)
    es, ee, trl, bl = metadata.define_blocks(trigs)
    return {'start': es, 'end': ee, 'trial': trl, 'block': bl}


def load_block(raw, trials, block):
    '''
    Crop a block of trials from raw file.
    '''
    start = int(trials['start'][trials['block'] == block].min())
    end = int(trials['end'][trials['block'] == block].max())
    print(start, end)
    r = raw.copy().crop(
        max(0, raw.times[start] - 5), min(raw.times[-1], raw.times[end] + 5))
    r_id = {'first_samp': r.first_samp}
    r.load_data()
    return r, r_id


def get_meta(data, raw, snum, block, filename):
    '''
    Return meta and timing data for a raw file and align it with behavioral data.

    Parameters
    ----------
    data : DataFrame
        Contains trial meta data from behavioral responses.
    raw : mne.io.raw objects
        MEG data that needs to be aligned to the behavioral data.
    snum : int
        Subject number that this raw object corresponds to.
    block : int
        Block within recording that this raw object corresponds to.

    Note: Data is matched againts the behavioral data with snum, recording, trial
    number and block number. Since the block number is not encoded in MEG data it
    needs to be passed explicitly. The order of responses is encoded in behavioral
    data and MEG data and is compared to check for alignment.
    '''
    trigs, buts = pymegprepr.get_events(raw)
    es, ee, trl, bl = metadata.define_blocks(trigs)

    megmeta = metadata.get_meta(trigs, es, ee, trl, bl,
                                metadata.fname2session(filename), snum,
                                buttons=buts)
    assert len(np.unique(megmeta.snum) == 1)
    assert len(np.unique(megmeta.day) == 1)
    assert len(np.unique(megmeta.block_num) == 1)

    dq = data.query('snum==%i & day==%i & block_num==%i' %
                    (megmeta.snum.ix[0], megmeta.day.ix[0], block))
    # dq.loc[:, 'trial'] = data.loc[:, 'trial']
    trial_idx = np.in1d(dq.trial, np.unique(megmeta.trial))
    dq = dq.iloc[trial_idx, :]
    dq = dq.set_index(['day', 'block_num', 'trial'])
    megmeta = metadata.correct_recording_errors(megmeta)
    megmeta.loc[:, 'block_num'] = block

    megmeta = megmeta.set_index(['day', 'block_num', 'trial'])
    del megmeta['snum']
    meta = pd.concat([megmeta, dq], axis=1)
    meta = metadata.cleanup(meta)  # Performs some alignment checks
    cols = [x for x in meta.columns if x[-2:] == '_t']
    timing = meta.loc[:, cols]
    return meta.drop(cols, axis=1), timing


#@memory.cache
def get_epochs_for_subject(snum, epoch, sessions=None):
    from itertools import product

    if sessions is None:
        sessions = list(range(4))
    data = []
    meta = get_meta_for_subject(snum, epoch, sessions=sessions)
    for session, block in product(ensure_iter(sessions), list(range(5))):
        filename = metadata.get_epoch_filename(
            snum, session, block, epoch, 'fif')
        if os.path.isfile(filename):
            data.append(filename)
    #assert len(data) == len(list(ensure_iter(sessions)))*5
    data = pymegprepr.load_epochs(data)
    event_ids = reduce(lambda x, y: x + y,
                       [list(d.event_id.values()) for d in data])
    meta = meta.reset_index().set_index('hash')
    meta = meta.loc[event_ids, :]
    assert len(meta) == sum([d._data.shape[0] for d in data])
    return pymegprepr.concatenate_epochs(data, [meta])


def get_meta_for_subject(snum, epoch, sessions=None):
    if sessions is None:
        sessions = list(range(5))
    metas = []
    for session, block in product(ensure_iter(sessions), list(range(5))):
        filename = metadata.get_epoch_filename(
            snum, session, block, epoch, 'meta')
        if os.path.isfile(filename + '.msgpack'):
            df = pd.read_msgpack(filename + '.msgpack')
            metas.append(df)
        elif os.path.isfile(filename):
            df = pd.read_hdf(filename)
            metas.append(df)
        else:
            pass
    #meta = pymegprepr.load_meta(metas)
    meta = pd.concat(metas).reset_index()
    cols = [c.decode('utf-8') if type(c) == bytes else c for c in meta.columns]
    meta.columns = cols
    return meta


@memory.cache
def get_head_correct_info(subject, session, N=-1):
    filename = metadata.get_raw_filename(subject, session)
    trans = get_ctf_trans(filename)
    fiducials = get_ref_head_pos(subject, session, trans, N=N)
    raw = mne.io.ctf.read_raw_ctf(filename)
    info = replace_fiducials(raw.info, fiducials)
    return trans, fiducials, info


def make_trans(subject, session):
    '''
    Create coregistration between MRI and MEG space.
    '''
    import os
    import time
    trans, fiducials, info = get_head_correct_info(subject, session)
    hs_ref = '/home/nwilming/conf_meg/trans/S%i-SESS%i.fif' % (
        subject, session)
    mne.io.meas_info.write_info(hs_ref, info)
    trans_name = ('/home/nwilming/conf_meg/trans/S%i-SESS%i-trans.fif' %
                  (subject, session))
    if os.path.isfile(trans_name):
        print('Removing previous trans file')
        os.remove(trans_name)
    print('--------------------------------')
    print('Please save trans file as:')
    print(trans_name)
    cmd = 'mne coreg --high-res-head -d %s -s %s -f %s' % (
        '/home/nwilming/fs_subject_dir', 'S%02i' % subject, hs_ref)
    print(cmd)
    os.system(cmd)
    # mne.gui.coregistration(inst=hs_ref, subject='S%02i' % subject,
    #                       subjects_dir='/home/nwilming/fs_subject_dir')
    while not os.path.isfile(trans_name):
        time.sleep(1)


@memory.cache
def get_ref_head_pos(subject, session,  trans, N=-1):
    from mne.transforms import apply_trans
    data = metadata.get_epoch_filename(subject, session, 0, 'stimulus', 'fif')
    data = pymegprepr.load_epochs([data])[0]
    cc = head_loc(data.decimate(10))
    nasion = np.stack([c[0] for c in cc[:N]]).mean(0)
    lpa = np.stack([c[1] for c in cc[:N]]).mean(0)
    rpa = np.stack([c[2] for c in cc[:N]]).mean(0)
    nasion, lpa, rpa = nasion.mean(-1), lpa.mean(-1), rpa.mean(-1)

    return {'nasion': apply_trans(trans['t_ctf_dev_dev'], np.array(nasion)),
            'lpa': apply_trans(trans['t_ctf_dev_dev'], np.array(lpa)),
            'rpa': apply_trans(trans['t_ctf_dev_dev'], np.array(rpa))}


def replace_fiducials(info, fiducials):
    from mne.io import meas_info
    fids = meas_info._make_dig_points(**fiducials)
    info = info.copy()
    dig = info['dig']
    for i, d in enumerate(dig):
        if d['kind'] == 3:
            if d['ident'] == 3:

                dig[i]['r'] = fids[2]['r']
            elif d['ident'] == 2:
                dig[i]['r'] = fids[1]['r']
            elif d['ident'] == 1:
                dig[i]['r'] = fids[0]['r']
    info['dig'] = dig
    return info


def head_movement(epochs):
    ch_names = np.array(epochs.ch_names)
    channels = {'x': ['HLC0011', 'HLC0012', 'HLC0013'],
                'y': ['HLC0021', 'HLC0022', 'HLC0023'],
                'z': ['HLC0031', 'HLC0032', 'HLC0033']}
    channel_ids = {}
    for key, names in channels.items():
        ids = [np.where([n in ch for ch in ch_names])[0][0] for n in names]
        channel_ids[key] = ids

    data = epochs._data
    ccs = []
    for e in range(epochs._data.shape[0]):
        x = np.stack([data[e, i, :] for i in channel_ids['x']])
        y = np.stack([data[e, i, :] for i in channel_ids['y']])
        z = np.stack([data[e, i, :] for i in channel_ids['z']])
        cc = circumcenter(x, y, z)
        ccs.append(cc)
    return np.stack(ccs)


@memory.cache
def get_head_loc(subject, session):
    epochs, meta = get_epochs_for_subject(
        subject, 'stimulus', sessions=[session])
    cc = head_loc(epochs)
    trans, fiducials, info = get_head_correct_info(subject, session)
    nose_coil = np.concatenate([c[0] for c in cc], -1)
    left_coil = np.concatenate([c[1] for c in cc], -1)
    right_coil = np.concatenate([c[2] for c in cc], -1)
    nose_coil = apply_trans(trans['t_ctf_dev_dev'], nose_coil.T)
    left_coil = apply_trans(trans['t_ctf_dev_dev'], left_coil.T)
    right_coil = apply_trans(trans['t_ctf_dev_dev'], right_coil.T)

    nose_coil = (nose_coil**2).sum(1)**.5
    left_coil = (left_coil**2).sum(1)**.5
    right_coil = (right_coil**2).sum(1)**.5
    return nose_coil, left_coil, right_coil


def head_loc(epochs):
    ch_names = np.array(epochs.ch_names)
    channels = {'x': ['HLC0011', 'HLC0012', 'HLC0013'],
                'y': ['HLC0021', 'HLC0022', 'HLC0023'],
                'z': ['HLC0031', 'HLC0032', 'HLC0033']}
    channel_ids = {}
    for key, names in channels.items():
        ids = [np.where([n in ch for ch in ch_names])[0][0] for n in names]
        channel_ids[key] = ids

    data = epochs._data
    ccs = []
    if len(epochs._data.shape) > 2:
        for e in range(epochs._data.shape[0]):
            x = np.stack([data[e, i, :] for i in channel_ids['x']])
            y = np.stack([data[e, i, :] for i in channel_ids['y']])
            z = np.stack([data[e, i, :] for i in channel_ids['z']])
            ccs.append((x, y, z))
    else:
        x = np.stack([data[i, :] for i in channel_ids['x']])
        y = np.stack([data[i, :] for i in channel_ids['y']])
        z = np.stack([data[i, :] for i in channel_ids['z']])
        ccs.append((x, y, z))
    return ccs


def get_ctf_trans(directory):
    from mne.io.ctf.res4 import _read_res4
    from mne.io.ctf.hc import _read_hc
    from mne.io.ctf.trans import _make_ctf_coord_trans_set

    res4 = _read_res4(directory)  # Read the magical res4 file
    coils = _read_hc(directory)  # Read the coil locations

    # Investigate the coil location data to get the coordinate trans
    coord_trans = _make_ctf_coord_trans_set(res4, coils)
    return coord_trans


def circumcenter(coil1, coil2, coil3):
    # Adapted from:
    #    http://www.fieldtriptoolbox.org/example/how_to_incorporate_head_movements_in_meg_analysis
    # CIRCUMCENTER determines the position and orientation of the circumcenter
    # of the three fiducial markers (MEG headposition coils).
    #
    # Input: X,y,z-coordinates of the 3 coils [3 X N],[3 X N],[3 X N] where N
    # is timesamples/trials.
    #
    # Output: X,y,z-coordinates of the circumcenter [1-3 X N], and the
    # orientations to the x,y,z-axes [4-6 X N].
    #
    # A. Stolk, 2012

    # number of timesamples/trials
    N = coil1.shape[1]
    cc = np.zeros((6, N)) * np.nan
    # x-, y-, and z-coordinates of the circumcenter
    # use coordinates relative to point `a' of the triangle
    xba = coil2[0, :] - coil1[0, :]
    yba = coil2[1, :] - coil1[1, :]
    zba = coil2[2, :] - coil1[2, :]
    xca = coil3[0, :] - coil1[0, :]
    yca = coil3[1, :] - coil1[1, :]
    zca = coil3[2, :] - coil1[2, :]

    # squares of lengths of the edges incident to `a'
    balength = xba * xba + yba * yba + zba * zba
    calength = xca * xca + yca * yca + zca * zca

    # cross product of these edges
    xcrossbc = yba * zca - yca * zba
    ycrossbc = zba * xca - zca * xba
    zcrossbc = xba * yca - xca * yba

    # calculate the denominator of the formulae
    denominator = 0.5 / (xcrossbc * xcrossbc + ycrossbc * ycrossbc
                         + zcrossbc * zcrossbc)

    # calculate offset (from `a') of circumcenter
    xcirca = ((balength * yca - calength * yba) * zcrossbc -
              (balength * zca - calength * zba) * ycrossbc) * denominator
    ycirca = ((balength * zca - calength * zba) * xcrossbc -
              (balength * xca - calength * xba) * zcrossbc) * denominator
    zcirca = ((balength * xca - calength * xba) * ycrossbc -
              (balength * yca - calength * yba) * xcrossbc) * denominator

    cc[0, :] = xcirca + coil1[0, :]
    cc[1, :] = ycirca + coil1[1, :]
    cc[2, :] = zcirca + coil1[2, :]
    # orientation of the circumcenter with respect to the x-, y-, and z-axis
    # coordinates
    v = np.stack([cc[0, :].T, cc[1, :].T, cc[2, :].T]).T
    vx = np.stack([np.zeros((N,)).T, cc[1, :].T, cc[2, :].T]).T
    # on the x - axis
    vy = np.stack([cc[0, :].T, np.zeros((N,)).T, cc[2, :].T]).T
    # on the y - axis
    vz = np.stack([cc[0, :].T, cc[1, :].T, np.zeros((N,)).T]).T
    # on the z - axis
    thetax, thetay = np.zeros((N,)) * np.nan, np.zeros((N,)) * np.nan
    thetaz = np.zeros((N,)) * np.nan
    for j in range(N):

        # find the angles of two vectors opposing the axes
        thetax[j] = np.arccos(np.dot(v[j, :], vx[j, :]) /
                              (np.linalg.norm(v[j, :]) * np.linalg.norm(vx[j, :])))
        thetay[j] = np.arccos(np.dot(v[j, :], vy[j, :]) /
                              (np.linalg.norm(v[j, :]) * np.linalg.norm(vy[j, :])))
        thetaz[j] = np.arccos(np.dot(v[j, :], vz[j, :]) /
                              (np.linalg.norm(v[j, :]) * np.linalg.norm(vz[j, :])))

        # convert to degrees
        cc[3, j] = (thetax[j] * (180 / np.pi))
        cc[4, j] = (thetay[j] * (180 / np.pi))
        cc[5, j] = (thetaz[j] * (180 / np.pi))
    return cc


def ensure_iter(input):
    if isinstance(input, str):
        yield input
    else:
        try:
            for item in input:
                yield item
        except TypeError:
            yield input
