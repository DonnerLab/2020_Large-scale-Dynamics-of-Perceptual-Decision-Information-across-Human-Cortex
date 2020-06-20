'''
Estimate signal and noise correlations between frequencies given
single sample gamma band power.
'''


import numpy as np
import pandas as pd

from conf_analysis.meg import fir_model as fm


def noise_corr(sa):
    '''
    Compute noise correlations: Estimate CRF, linearly interpolate 
    and get residuals
    '''
    from scipy.interpolate import interp1d
    k = fm.get_average_contrast(sa, by='F', centers=np.linspace(0, 1, 21))
    for F, df in k.groupby('F'):        
        func = interp1d(df.contrast.values, df.power)
        idx = sa.loc[:, 'F'] == F
        sa.loc[idx, 'residuals'] = sa.loc[idx, 'power']-func(sa.loc[idx, 'contrast'])

    kmean = fm.contrast_integrated_averages(sa, centers=np.linspace(0, 1, 21))

    func = interp1d(kmean.contrast.values, kmean.power)
    sa.loc[:, 'residuals_F_ind'] = sa.loc[:, 'power']-func(sa.loc[:, 'contrast'])
    return sa


def get_correlations(sa, contrast_cut=None):
    '''
    Compute noise and signal correlations across frequencies.
    '''
    accncorr = []
    accscorr = []
    for subject, dsub in sa.groupby('subject'):

        nc = noise_corr(dsub)
        if contrast_cut is not None:
            nc = nc.query('contrast %s' % contrast_cut)
        ncorr = pd.pivot_table(nc, values='residuals',
                               index='F', columns='sample_id')
        accncorr.append(
            np.corrcoef(ncorr)
        )

        scorr = fm.get_average_contrast(
            dsub, by=['F'], centers=np.linspace(0, 1, 21))
        if contrast_cut is not None:
            scorr = scorr.query('contrast %s' % contrast_cut)

        sk = pd.pivot_table(scorr, values='power',
                            index='F', columns='contrast')
        accscorr.append(
            np.corrcoef(sk)
        )
    return np.stack(accscorr), np.stack(accncorr)


def plot_corrs(signal, noise, gs=None, offset=0, vmin=0, cmap='viridis', tri_func=np.tril):
    from matplotlib import gridspec
    import pylab as plt
    import seaborn as sns
    if gs is None:
        gs = gridspec.GridSpec(1, 2)
    sns.set_style('ticks')
    plt.subplot(gs[offset, 0])
    plt.imshow(tri_func(signal.mean(0)), interpolation='nearest', aspect='auto',
               cmap=cmap, vmin=vmin, vmax=1)
    plt.yticks(range(7), range(40, 75, 5))
    plt.xticks(range(7), range(40, 75, 5))
    plt.xlabel('F')
    plt.ylabel('F')
    plt.title('Signal correlations')
    sns.despine(ax=plt.gca())
    plt.subplot(gs[offset, 1])
    plt.imshow(tri_func(noise.mean(0)), interpolation='nearest', aspect='auto',
               cmap=cmap, vmin=vmin, vmax=1)
    plt.yticks(range(7), range(40, 75, 5))
    plt.xticks(range(7), range(40, 75, 5))
    plt.xlabel('F')
    plt.ylabel('F')
    plt.title('Noise correlations')
    sns.despine(ax=plt.gca())
    plt.subplot(gs[offset, 0])
    #plt.colorbar()
    plt.subplot(gs[offset, 1])
    #plt.colorbar()
