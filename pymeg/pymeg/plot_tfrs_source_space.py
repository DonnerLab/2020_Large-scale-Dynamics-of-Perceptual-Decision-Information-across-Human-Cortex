import os
import glob
import numpy as np
import scipy as sp
import pandas as pd
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
from joblib import Parallel, delayed
from joblib import Memory

from IPython import embed as shell

import pymeg
from pymeg import preprocessing as prep
from pymeg import tfr as tfr
from pymeg import atlas_glasser
from pymeg import contrast_tfr

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(style='ticks', font='Arial', font_scale=1, rc={
    'axes.linewidth': 0.25, 
    'axes.labelsize': 7, 
    'axes.titlesize': 7, 
    'xtick.labelsize': 6, 
    'ytick.labelsize': 6, 
    'legend.fontsize': 7, 
    'xtick.major.width': 0.25, 
    'ytick.major.width': 0.25,
    'text.color': 'Black',
    'axes.labelcolor':'Black',
    'xtick.color':'Black',
    'ytick.color':'Black',} )
sns.plotting_context()

memory = Memory(cachedir=os.environ['PYMEG_CACHE_DIR'], verbose=0)

# data_folder = '/home/jw/share/data/'
# fig_folder = '/home/jw/share/figures/'
data_folder = 'Z:\\JW\\data'
fig_folder = 'Z:\\JW\\figures'
# data_folder = '/home/jwdegee/degee/MEG/data/'
# fig_folder = '/home/jwdegee/degee/MEG/figures/'

def load_meta_data(subj, session, timelock, data_folder):

    # load:
    meta_data_filename = os.path.join(data_folder, "epochs", subj, session, '{}-meta.hdf'.format(timelock))
    meta_data = pymeg.preprocessing.load_meta([meta_data_filename])[0]

    # add columns:
    meta_data["all"] = 1
    meta_data["left"] = (meta_data["resp_meg"] < 0).astype(int)
    meta_data["right"] = (meta_data["resp_meg"] > 0).astype(int)
    meta_data["hit"] = ((meta_data["stimulus"] == 1) & (meta_data["choice_a"] == 1)).astype(int)
    meta_data["fa"] = ((meta_data["stimulus"] == 0) & (meta_data["choice_a"] == 1)).astype(int)
    meta_data["miss"] = ((meta_data["stimulus"] == 1) & (meta_data["choice_a"] == 0)).astype(int)
    meta_data["cr"] = ((meta_data["stimulus"] == 0) & (meta_data["choice_a"] == 0)).astype(int)
    meta_data["left"] = (meta_data["resp_meg"] < 0).astype(int)
    meta_data["right"] = (meta_data["resp_meg"] > 0).astype(int)
    meta_data["pupil_h"] = (meta_data["pupil_lp_d"] >= np.percentile(meta_data["pupil_lp_d"], 60)).astype(int)
    meta_data["pupil_l"] = (meta_data["pupil_lp_d"] <= np.percentile(meta_data["pupil_lp_d"], 40)).astype(int)
    return meta_data

def compute_contrasts(subj, sessions, contrasts, hemis, baseline_time=(-0.25, -0.15), n_jobs=1):

    tfrs_stim = []
    tfrs_resp = []
    for session in sessions:
        with contrast_tfr.Cache() as cache:
            for timelock in ['stimlock', 'resplock']:
                
                data_globstring = os.path.join(data_folder, "source_level", "lcmv_{}_{}_{}_*-source.hdf".format(subj, session, timelock,))
                base_globstring = os.path.join(data_folder, "source_level", "lcmv_{}_{}_{}_*-source.hdf".format(subj, session, 'stimlock',))
                
                # tfrs = []
                # tfr_data_filenames = glob.glob(data_globstring)
                # for f in tfr_data_filenames:
                #     tfr = pd.read_hdf(f)
                #     tfr = pd.pivot_table(tfr.reset_index(), values=tfr.columns, index=[
                #                  'trial', 'est_val'], columns='time').stack(-2)
                #     tfr.index.names = ['trial', 'freq', 'area']
                #     tfrs.append(tfr)
                # tfr = pd.concat(tfrs)

                meta_data = load_meta_data(subj, session, timelock, data_folder)
                tfr = contrast_tfr.compute_contrast(contrasts, hemis, data_globstring, base_globstring, 
                                                        meta_data, baseline_time, n_jobs=n_jobs, cache=cache)
                tfr['subj'] = subj
                tfr['session'] = session
                tfr = tfr.set_index(['cluster', 'subj', 'session', 'contrast'], append=True, inplace=False)
                tfr = tfr.reorder_levels(['subj', 'session', 'cluster', 'contrast', 'freq'])
                if timelock == 'stimlock':
                    tfrs_stim.append(tfr)
                elif timelock == 'resplock':
                    tfrs_resp.append(tfr)
    tfrs_stim = pd.concat(tfrs_stim)
    tfrs_resp = pd.concat(tfrs_resp)

    # mean across sessions:
    tfrs_stim = tfrs_stim.groupby(['subj', 'cluster', 'contrast', 'freq']).mean()
    tfrs_resp = tfrs_resp.groupby(['subj', 'cluster', 'contrast', 'freq']).mean()

    # save:
    tfrs_stim.to_hdf(os.path.join(data_folder, 'source_level', 'contrasts', 'tfr_contrasts_stimlock_{}.hdf'.format(subj)), 'tfr')
    tfrs_resp.to_hdf(os.path.join(data_folder, 'source_level', 'contrasts', 'tfr_contrasts_resplock_{}.hdf'.format(subj)), 'tfr')

def load_contrasts(subj, timelock):
    tfr = pd.read_hdf(os.path.join(data_folder, 'source_level', 'contrasts', 'tfr_contrasts_{}_{}.hdf'.format(timelock, subj)))
    return tfr

def plot_tfr(tfr, time_cutoff, vmin, vmax, tl, cluster_correct=False, threshold=0.05, ax=None):

    # colorbar:
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'lightblue', 'lightgrey', 'yellow', 'red'], N=100)
    
    # variables:
    times = np.array(tfr.columns, dtype=float)
    freqs = np.array(np.unique(tfr.index.get_level_values('freq')), dtype=float)
    time_ind = (times>time_cutoff[0]) & (times<time_cutoff[1])
    time_ind = (times>time_cutoff[0]) & (times<time_cutoff[1])
    
    # data:
    X = np.stack([tfr.loc[tfr.index.isin([subj], level='subj'), time_ind].values for subj in np.unique(tfr.index.get_level_values('subj'))])
    
    # grand average plot:
    cax = ax.pcolormesh(times[time_ind], freqs, X.mean(axis=0), vmin=vmin, vmax=vmax, cmap=cmap)
        
    # cluster stats:
    if cluster_correct:
        if tl  == 'stimlock':
            test_data = X[:,:,times[time_ind]>0]
            times_test_data = times[time_ind][times[time_ind]>0]
        else:
            test_data = X.copy()
            times_test_data = times[time_ind]
        try:
            T_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_1samp_test(test_data, threshold={'start':0, 'step':0.2}, 
                                                                                        connectivity=None, tail=0, n_permutations=1000, n_jobs=10)
            sig = cluster_p_values.reshape((test_data.shape[1], test_data.shape[2]))
            ax.contour(times_test_data, freqs, sig, (threshold,), linewidths=0.5, colors=('black'))
        except:
            pass

    ax.axvline(0, ls='--', lw=0.75, color='black',)
    plt.colorbar(cax, ticks=[vmin, 0, vmax])

    return ax

if __name__ == '__main__':
    
    compute = True
    plot = True

    # subjects:
    subjects = ['jw01', 'jw02', 'jw03', 'jw05', 'jw07', 'jw08', 'jw09', 'jw10', 'jw11', 'jw12', 'jw13', 'jw14', 'jw15', 'jw16', 'jw17', 'jw18', 'jw19', 'jw20', 'jw21', 'jw22', 'jw23', 'jw24', 'jw30']
    # subjects = ['jw01', 'jw02',]

    # get clusters:
    all_clusters, visual_field_clusters, glasser_clusters, jwg_clusters = atlas_glasser.get_clusters()

    # define contrasts:
    contrasts = {
    'all': (['all'], [1]),
    'choice': (['hit', 'fa', 'miss', 'cr'], (1, 1, -1, -1)),
    'stimulus': (['hit', 'fa', 'miss', 'cr'], (1, -1, 1, -1)),
    'hand': (['left', 'right'], (1, -1)),
    'pupil': (['pupil_h', 'pupil_l'], (1, -1)),
    }

    # hemis:
    hemis = ['avg', 'avg', 'avg', 'rh_is_ipsi', 'avg']
    
    # compute contrasts:
    if compute:
        for subj in subjects:
            if subj == 'jw16':
                sessions = ['B']
            elif subj == 'jw24':
                sessions = ['A']
            elif subj == 'jw30':
                sessions = ['A']
            else:
                sessions = ['A', 'B']
            compute_contrasts(subj, sessions, contrasts, hemis)
    
    # plot:
    if plot:
        for timelock in ['stimlock', 'resplock']:
            tfrs = pd.concat([load_contrasts(subj, timelock) for subj in subjects])
            for contrast_name in contrasts.keys():
                for cluster in all_clusters.keys():

                    # tfr to plot:
                    tfr = tfrs.loc[tfrs.index.isin([cluster], level='cluster') & 
                                        tfrs.index.isin([contrast_name], level='contrast')]

                    # time:
                    if timelock  == 'stimlock':
                        time_cutoff = (-0.35, 1.3)
                        xlabel = 'Time from stimulus (s)'
                    if timelock  == 'resplock':
                        time_cutoff = (-0.7, 0.2)
                        xlabel = 'Time from response (s)'

                    # vmin vmax:
                    if contrast_name == 'all':
                        vmin, vmax = (-15, 15)
                    else:
                        vmin, vmax = (-10, 10)

                    fig = plt.figure(figsize=(2,2))
                    ax = fig.add_subplot(111)
                    plot_tfr(tfr, time_cutoff, vmin, vmax, timelock, 
                                cluster_correct=True, threshold=0.05, ax=ax)
                    ax.set_title('{} contrast (N={})'.format(contrast_name, len(subjects)))
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel('Frequency (Hz)')
                    if timelock == 'stimlock':
                        ax.axvline(-0.25, ls=':', lw=0.75, color='black',)
                        ax.axvline(-0.15, ls=':', lw=0.75, color='black',)
                    elif timelock == 'resplock':
                        ax.tick_params(labelleft='off')
                    fig.tight_layout()
                    fig.savefig(os.path.join(fig_folder, 'source_level', '{}_{}_{}.pdf'.format(cluster, contrast_name, timelock)))