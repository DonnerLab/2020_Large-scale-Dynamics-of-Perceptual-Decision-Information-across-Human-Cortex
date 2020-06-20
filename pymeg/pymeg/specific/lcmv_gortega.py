import logging
import mne
import numpy as np
import os

from joblib import Memory

from os import makedirs
from os.path import join
from glob import glob

from pymeg import lcmv
from pymeg import preprocessing
from pymeg import source_reconstruction as sr
import pandas as pd
import pickle

memory = Memory(cachedir=os.environ['PYMEG_CACHE_DIR'])
#path = '/home/gprat/cluster_archive/Master_Project/preprocessed_megdata'
path = '/mnt/archive/genis/Master_Project/preprocessed_megdata'


def set_n_threads(n):
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = str(n)
    os.environ['MKL_NUM_THREADS'] = str(n)
    os.environ['OMP_NUM_THREADS'] = str(n)


subjects = {}


def submit():
    from pymeg import parallel
    for subject, tasks in subjects.items():
        for session, recording in tasks:
            for signal in ['BB', 'HF', 'LF']:
                parallel.pmap(
                    extract, [(subject, session, recording, signal)],
                    walltime='15:00:00', memory=50, nodes=1, tasks=4,
                    name='SR' + str(subject) + '_' +
                    str(session) + str(recording),
                    ssh_to=None, env='mne')


def lcmvfilename(subject, session, signal, recording, epoch, chunk=None):
    try:
        makedirs(path)
    except:
        pass
    if chunk is None:
        filename = '%s-SESS%i-%s-%i-%s-lcmv.hdf' % (
            subject, session, epoch, recording, signal)
    else:
        filename = '%s-SESS%i-%s-%i-%s-chunk%i-lcmv.hdf' % (
            subject, session, epoch, recording, signal, chunk)
    return join(path, filename)


def get_filenames(subject, session, recording):
    fname='/mnt/archive/genis/Master_Project/preprocessed_megdata/filenames_sub%i.pickle'% (subject)
    f=open(fname,'rb')
    data=pickle.load(f)
    df=pd.DataFrame.from_dict(data)
    blocks=df[df.subject==subject][df.session==session][df.trans_matrix==recording].block
    stimulus=[]
    response=[]
    for b in blocks:
        stimulus.append( path+'/stim_meta_sub%i_sess%i_block%i_offset*-epo.fif.gz' % (subject, session,b))
        response.append( path+'/resp_meta_sub%i_sess%i_block%i_offset*-epo.fif.gz' % (subject, session,b))              



#    if recording == 1:
#        stimulus = join(path, 'stim_meta_sub%i_sess%i_block[1-3]_offset*-epo.fif.gz' % (
#            subject, session))
#        response = join(path, 'resp_meta_sub%i_sess%i_block[1-3]_offset*-epo.fif.gz' % (
#            subject, session))
#    elif recording == 2:
#        stimulus = join(path, 'stim_meta_sub%i_sess%i_block[4-6]_offset*-epo.fif.gz' % (
#            subject, session))
#        response = join(path, 'resp_meta_sub%i_sess%i_block[4-6]_offset*-epo.fif.gz' % (
#            subject, session))
#    elif recording == 3:
#        stimulus = join(path, 'stim_meta_sub%i_sess%i_block[7-9]_offset*-epo.fif.gz' % (
#            subject, session))
#        response = join(path, 'resp_meta_sub%i_sess%i_block[7-9]_offset*-epo.fif.gz' % (
#            subject, session))
    return stimulus, response


def get_stim_epoch(subject, session, recording):
    filenames = glob(get_filenames(subject, session, recording)[0])
    epochs = preprocessing.load_epochs(filenames)
    epochs=preprocessing.concatenate_epochs(epochs,None)
    epochs = epochs.pick_channels(
        [x for x in epochs.ch_names if x.startswith('M')])

    id_time = (-0.25 <= epochs.times) & (epochs.times <= 0)
    means = epochs._data[:, :, id_time].mean(-1)
    epochs._data -= means[:, :, np.newaxis]
    data_cov = lcmv.get_cov(epochs, tmin=0, tmax=3)
    return data_cov, epochs, filenames


def get_response_epoch(subject, session, recording):
    stimulus = glob(get_filenames(subject, session, recording)[0])
    response = glob(get_filenames(subject, session, recording)[1])
    
    stimulus = preprocessing.load_epochs(stimulus)
    stimulus=preprocessing.concatenate_epochs(stimulus,None)
    stimulus = stimulus.pick_channels(
        [x for x in stimulus.ch_names if x.startswith('M')])
    response = preprocessing.load_epochs(response)
    response=preprocessing.concatenate_epochs(response,None)
    response = stimulus.pick_channels(
        [x for x in response.ch_names if x.startswith('M')])

    id_time = (-0.25 <= stimulus.times) & (stimulus.times <= 0)
    means = stimulus._data[:, :, id_time].mean(-1)
    stimulus._data = stimulus._data - means[:, :, np.newaxis]
    response._data = response._data - means[:, :, np.newaxis]
    data_cov = lcmv.get_cov(stimulus, tmin=0, tmax=3)
    return data_cov, response


def extract(subject, session, recording, epoch, signal_type='BB',
            BEM='three_layer', debug=False, chunks=100, njobs=4):
    mne.set_log_level('WARNING')
    lcmv.logging.getLogger().setLevel(logging.INFO)
    set_n_threads(1)

    logging.info('Reading stimulus data')

    if epoch == 'stimulus':
        data_cov, epochs, epochs_filename = get_stim_epoch(
            subject, session, recording)
    else:
        data_cov, epochs, epochs_filename = get_response_epoch(
            subject, session, recording)

    raw_filename = glob('TODO' % (subject, session, recording))

    trans_filename = glob('TODO' % (subject, session, recording))[0]
    logging.info('Setting up source space and forward model')

    forward, bem, source = sr.get_leadfield(
        subject, raw_filename, epochs_filename, trans_filename,
        bem_sub_path='bem_ft')
    labels = sr.get_labels(subject)
    labels = sr.labels_exclude(labels, exclude_filters=['wang2015atlas.IPS4',
                                                        'wang2015atlas.IPS5',
                                                        'wang2015atlas.SPL',
                                                        'JWDG_lat_Unknown'])
    labels = sr.labels_remove_overlap(
        labels, priority_filters=['wang', 'JWDG'],)

    fois_h = np.arange(36, 162, 4)
    fois_l = np.arange(2, 36, 1)
    tfr_params = {
        'HF': {'foi': fois_h, 'cycles': fois_h * 0.25, 'time_bandwidth': 2 + 1,
               'n_jobs': njobs, 'est_val': fois_h, 'est_key': 'HF', 'sf': 600,
               'decim': 10},
        'LF': {'foi': fois_l, 'cycles': fois_l * 0.4, 'time_bandwidth': 1 + 1,
               'n_jobs': njobs, 'est_val': fois_l, 'est_key': 'LF', 'sf': 600,
               'decim': 10}
    }

    events = epochs.events[:, 2]
    filters = lcmv.setup_filters(epochs.info, forward, data_cov,
                                 None, labels)
    set_n_threads(1)

    for i in range(0, len(events), chunks):
        filename = lcmvfilename(
            subject, session, signal_type, recording, chunk=i)
        if os.path.isfile(filename):
            continue
        if signal_type == 'BB':
            logging.info('Starting reconstruction of BB signal')
            M = lcmv.reconstruct_broadband(
                filters, epochs.info, epochs._data[i:i + chunks],
                events[i:i + chunks],
                epochs.times, njobs=1)
        else:
            logging.info('Starting reconstruction of TFR signal')
            M = lcmv.reconstruct_tfr(
                filters, epochs.info, epochs._data[i:i + chunks],
                events[i:i + chunks], epochs.times,
                est_args=tfr_params[signal_type],
                njobs=4)
        M.to_hdf(filename, 'epochs')
    set_n_threads(njobs)
