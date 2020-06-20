import os
import glob
import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib

from IPython import embed as shell

import pymeg
from pymeg import preprocessing as prep
from pymeg import source_reconstruction as sr
from pymeg import lcmv 

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(style='ticks', font='Arial', font_scale=1, rc={
    'axes.linewidth': 0.25, 
    'axes.labelsize': 7, 
    'axes.titlesize': 7, 
    'xtick.labelsize': 6, 
    'ytick.labelsize': 6, 
    'legend.fontsize': 6, 
    'xtick.major.width': 0.25, 
    'ytick.major.width': 0.25,
    'text.color': 'Black',
    'axes.labelcolor':'Black',
    'xtick.color':'Black',
    'ytick.color':'Black',} )
sns.plotting_context()

data_folder = 'Z:\\JW\\data'
sr.set_fs_subjects_dir('Z:\\JW\\data\\fs_subjects')
# data_folder = '/home/jw/share/data/'
# sr.set_fs_subjects_dir('/home/jw/share/data/fs_subjects/')

def get_glasser_labels(subj, subjects_dir='/home/jw/share/data/fs_subjects/'):

    from pymeg import atlas_glasser as gl
    subjects = [subj]
    gl.get_hcp(subjects_dir)
    gl.get_hcp_annotation(subjects_dir, subj)
    
def get_broadband_estimator():
    return ('BB', [-1], lambda x: x[:, np.newaxis, :])

def do_source_recon(subj, session, njobs=4):
        
        runs = sorted([run.split('/')[-1] for run in glob.glob(os.path.join(data_folder, "raw", subj, session, "meg", "*.ds"))])
        center = int(np.floor(len(runs) / 2.0))
        raw_filename = os.path.join(data_folder, "raw", subj, session, "meg", runs[center])
        
        # # make transformation matrix:
        # sr.make_trans(subj, raw_filename, epochs_filename, trans_filename)

        epochs_filename_stim = os.path.join(data_folder, "epochs", subj, session, '{}-epo.fif.gz'.format('stimlock'))
        epochs_filename_resp = os.path.join(data_folder, "epochs", subj, session, '{}-epo.fif.gz'.format('resplock'))
        trans_filename = os.path.join(data_folder, "transformation_matrix", '{}_{}-trans.fif'.format(subj, session))

        if os.path.isfile(epochs_filename_stim):
            
            # load labels:
            labels = sr.get_labels(subject=subj, filters=['*wang*.label', '*JWG*.label'], annotations=['HCPMMP1'] )
            labels = sr.labels_exclude(labels=labels, exclude_filters=['wang2015atlas.IPS4', 'wang2015atlas.IPS5', 
                                                            'wang2015atlas.SPL', 'JWG_lat_Unknown'])
            labels = sr.labels_remove_overlap(labels=labels, priority_filters=['wang', 'JWG'])

            # load epochs:
            epochs_stim = mne.read_epochs(epochs_filename_stim)
            epochs_stim = epochs_stim.pick_channels([x for x in epochs_stim.ch_names if x.startswith('M')])
            epochs_resp = mne.read_epochs(epochs_filename_resp)
            epochs_resp = epochs_resp.pick_channels([x for x in epochs_resp.ch_names if x.startswith('M')])
            
            # baseline stuff:
            overlap = list(
                set(epochs_stim.events[:, 2]).intersection(
                set(epochs_resp.events[:, 2])))
            epochs_stim = epochs_stim[[str(l) for l in overlap]]
            epochs_resp = epochs_resp[[str(l) for l in overlap]]
            id_time = (-0.3 <= epochs_stim.times) & (epochs_stim.times <= -0.2)
            means = epochs_stim._data[:, :, id_time].mean(-1)
            epochs_stim._data = epochs_stim._data - means[:, :, np.newaxis]
            epochs_resp._data = epochs_resp._data - means[:, :, np.newaxis]
            
            # TFR settings:
            fois_h = np.arange(42, 162, 4)
            fois_l = np.arange(2, 42, 2)
            tfr_params = {
                'HF': {'foi': fois_h, 'cycles': fois_h * 0.4, 'time_bandwidth': 5+1,
                'n_jobs': njobs, 'est_val': fois_h, 'est_key': 'HF'},
                'LF': {'foi': fois_l, 'cycles': fois_l * 0.4, 'time_bandwidth': 1+1,
                'n_jobs': njobs, 'est_val': fois_l, 'est_key': 'LF'}
            }

            # get cov:
            data_cov = lcmv.get_cov(epochs_stim, tmin=0, tmax=1)
            noise_cov = None

            # get lead field:
            forward, bem, source = sr.get_leadfield(
                                                    subject=subj, 
                                                    raw_filename=raw_filename, 
                                                    epochs_filename=epochs_filename_stim, 
                                                    trans_filename=trans_filename,
                                                    conductivity=(0.3, 0.006, 0.3),
                                                    njobs=njobs
                                                    )

                        
            # do source level analysis:
            for tl, epochs in zip(['stimlock', 'resplock'], [epochs_stim, epochs_resp]):
                for signal_type in ['LF', 'HF']:
                    print(signal_type)

                    # events:
                    events = epochs.events[:, 2]
                    data = []
                    filters = lcmv.setup_filters(epochs.info, forward, data_cov,
                                          None, labels, njobs=njobs)
                    
                    # in chunks:
                    chunks = 100
                    for i in range(0, len(events), chunks):
                        filename = os.path.join(data_folder, "source_level", 'lcmv_{}_{}_{}_{}_{}-source.hdf'.\
                            format(subj, session, tl, signal_type, i))
                        # if os.path.isfile(filename):
                        #     continue
                        M = lcmv.reconstruct_tfr(
                            filters, epochs.info, epochs._data[i:i + chunks],
                            events[i:i + chunks], epochs.times,
                            est_args=tfr_params[signal_type],
                            njobs=njobs)
                        M.to_hdf(filename, 'epochs')


if __name__ == "__main__":
                    
    # subjects = ['jw01', 'jw02', 'jw03', 'jw05', 'jw07', 'jw08', 'jw09', 'jw10', 'jw11', 'jw12', 'jw13', 'jw14', 'jw15', 'jw16', 'jw17', 'jw18', 'jw19', 'jw20', 'jw21', 'jw22', 'jw23', 'jw24', 'jw30']
    subjects = ['jw08', 'jw09', 'jw10', 'jw11', 'jw12', 'jw13', 'jw14', 'jw15', 'jw16', 'jw17', 'jw18', 'jw19', 'jw20', 'jw21', 'jw22', 'jw23', 'jw24', 'jw30']
    for subj in subjects:
        # get_glasser_labels(subj)
        for session in ['A', 'B']:
            do_source_recon(subj, session, njobs=24)
               