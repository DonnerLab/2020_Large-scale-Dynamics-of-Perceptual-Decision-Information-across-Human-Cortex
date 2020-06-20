import logging
import os
import numpy as np
import pandas as pd

from os.path import join
from glob import glob

from pymeg import aggregate_sr as asr
from conf_analysis.behavior import metadata
from conf_analysis.behavior import empirical, kernels
from conf_analysis.meg import regress
from conf_analysis.meg import decoding_plots as dp, decoding_analysis as da
from scipy.stats import linregress

try:
    import pylab as plt
except:
    plt = None


from joblib import Memory

if "TMPDIR" in os.environ.keys():
    memory = Memory(cachedir=os.environ["PYMEG_CACHE_DIR"], verbose=0)
    inpath = "/nfs/nwilming/MEG/sr_labeled/aggs"
    outpath = "/nfs/nwilming/MEG/sr_decoding/"
elif "RRZ_LOCAL_TMPDIR" in os.environ.keys():
    tmpdir = os.environ["RRZ_LOCAL_TMPDIR"]
    outpath = "/work/faty014/MEG/sr_labeled/aggs/"
    outpath = "/work/faty014/MEG/sr_decoding/"
    memory = Memory(cachedir=tmpdir)
else:
    inpath = "/home/nwilming/conf_meg/sr_labeled/aggs"
    outpath = "/home/nwilming/conf_meg/sr_decoding"
    memory = Memory(cachedir=metadata.cachedir)


def prepare_and_save_data(freq_band=[45, 65], n_jobs=4, filtervar=None,
    outpath='/home/nwilming/conf_analysis/results/cort_kernel.results.pickle',):
    from joblib import Parallel, delayed
    from conf_analysis.meg import srtfr
    peaks = get_decoding_peaks()
    ogl_areas = list(srtfr.get_ogl_clusters().keys())
    ccs = Parallel(n_jobs=4)(
        delayed(correlate)(
            subject,
            "Averaged",
            cluster,
            0.19,
            peakslope=peaks,
            freq_band=freq_band,
            ogl=True,
        )
        for subject in range(1, 16)
        for cluster in ogl_areas
    )
    ccs = pd.DataFrame(ccs)

    rems = []
    alls = []
    nosplit_rems = []
    nosplit_alls = []
    from conf_analysis.meg import srtfr
    clusters = [r for r in srtfr.get_clusters().keys() if not r.startswith("NSW")]
    from itertools import product
    tasks = []
    for cluster, subject in product(clusters, np.arange(1, 16)):
        tasks.append(delayed(d_gck)(
            subject, "Averaged", cluster, 0.19, freq_band=freq_band, ogl=False,
            remove_contrast_induced_flucts=False, filtervar=filtervar,
            )
        )
        tasks.append(delayed(d_gck)(
            subject, "Averaged", cluster, 0.19, freq_band=freq_band, ogl=False, 
            remove_contrast_induced_flucts=True, filtervar=filtervar,
            )
        )
    results = Parallel(n_jobs=n_jobs)(tasks)
    
   
    K, C = get_kernels()
    import pickle
    kernels = pd.DataFrame(results)
    pickle.dump({'ccs':ccs, 'kernels':kernels,  'K':K, 'v1decoding':peaks}, open(outpath, 'wb'))
    return ccs, alls, rems, peaks, K


def to_fluct(Xs, side):
    ids = side == 1
    mean_larger = Xs[ids, :].mean(0)[np.newaxis, :]
    mean_smalle = Xs[~ids, :].mean(0)[np.newaxis, :]
    Xs[ids, :] -= mean_larger
    Xs[~ids, :] -= mean_larger
    return Xs


def get_kernel(subject):
    K, C = get_kernels()
    
    return K.loc[subject, :], C.loc[subject, :]


@memory.cache()
def get_kernels(d=None):
    
    if d is None:
        d = empirical.load_data()
    K = (
        d.groupby(["snum", "side"])
        .apply(kernels.get_decision_kernel)
        .groupby("snum")
        .mean()
    )  # dp.extract_kernels(dz)
    
    C = (
        d.groupby(["snum", "side"])
        .apply(kernels.get_confidence_kernel)
        .stack()
        .groupby("snum")
        .mean()
    )  # dp.extract_kernels(dz)
    print(C)
    return K, C

def kernel_by_RT(data, edges):
    from conf_analysis.meg import figures
    P = []
    for rt, d in data.groupby(pd.cut(data.choice_rt, edges)):
        K, _= get_kernels(d=d)
        figures.plotwerr(K, label=rt.mid)
        K.loc[:, 'rt'] = rt.mid
        P.append(K.set_index('rt', append=True))
    return pd.concat(P)

def _par_get_cortical_kernel(*args, **kwargs):
    return get_cortical_kernel(*args, **kwargs)


def d_gck(subject,
    hemi,
    cluster,
    latency,
    remove_contrast_induced_flucts=False,
    use_only_contrast_induced_flucts=False,
    freq_band=[45, 65],
    ogl=True,
    split_by_mc=True,
    filtervar=None):
    
    kernel = get_cortical_kernel(subject, hemi, cluster, latency,
        remove_contrast_induced_flucts=remove_contrast_induced_flucts,
        use_only_contrast_induced_flucts=use_only_contrast_induced_flucts,
        freq_band=freq_band,
        ogl=ogl,
        split_by_mc=split_by_mc,
        filtervar=filtervar)

    return {
        'cluster': cluster,
        'subject': subject,
        'kernel': kernel,
        'freq': freq_band,
        'rmcif': remove_contrast_induced_flucts,
        'filtervar': filtervar
    }


@memory.cache()
def get_cortical_kernel(
    subject,
    hemi,
    cluster,
    latency,
    remove_contrast_induced_flucts=False,
    use_only_contrast_induced_flucts=False,
    freq_band=[45, 65],
    ogl=True,
    split_by_mc=True,
    filtervar=None,
):
    """
    Computes how well fluctuations can separate responses for
    each contrast conditions. A contrast condition is either
    mean contrast > 0.5 or mean contrast < 0.5. The AUC
    values for each contrast condition are averaged. 
    """
    X, tpc, freq, meta = build_design_matrix(
        subject, cluster, latency, hemi, freq_bands=freq_band, ogl=ogl
    )
    if filtervar is not None:
        print('FILTERING to ', filtervar)
        idvar = meta.loc[:, "noise_sigma"] == filtervar
        X = X[idvar]
        meta = meta.loc[idvar]
        print(meta.shape)

    if remove_contrast_induced_flucts:
        contrast = np.stack(meta.contrast_probe)
        X = X - get_contrast_induced_fluctuations(X, contrast)
    if use_only_contrast_induced_flucts:
        contrast = np.stack(meta.contrast_probe)
        X = get_power_contrast_prediction(X, contrast)
    if split_by_mc:
        idside = meta.loc[:, "side"] == 1
        a = kernels.kernel(X[idside], meta.loc[idside, "response"])[0] - 0.5
        b = kernels.kernel(X[~idside], meta.loc[~idside, "response"])[0] - 0.5
        return (a + b) / 2
    else:
        return kernels.kernel(X[:], meta.loc[:, "response"])[0] - 0.5


def get_contrast_induced_fluctuations(data, contrast):
    """
    Predict contrast induced fluctuations by means of linear regression.

    Data is n_trials x 10 matrix of power values
    contrast is n_trials x 10 matrix of contrast values

    """
    data = data.copy()
    for i in range(10):
        slope, intercept, _, _, _ = linregress(contrast[:, i], data[:, i])
        data[:, i] = slope * contrast[:, i] + intercept
    return data


def get_power_contrast_prediction(data, contrast):
    """
    Predict contrast induced fluctuations by means of linear regression.

    Data is n_trials x 10 matrix of power values
    contrast is n_trials x 10 matrix of contrast values

    """
    data = data.copy()
    for i in range(10):
        slope, intercept, _, _, _ = linregress(data[:, i], contrast[:, i])
        data[:, i] = slope * data[:, i] + intercept
    return data


@memory.cache
def get_decoding_peaks():
    ssd = dp.get_ssd_data(ogl=True, restrict=False)
    return dp.extract_peak_slope(ssd).test_slope.Pair


def correlate(
    subject, hemi, cluster, latency, peakslope=None, freq_band=[45, 65], ogl=False
):
    """
    Compute three different correlations with choice kernel:

    1) Contrast induced fluctuations
    This measures the profile of contrast decodability and correlates
    it with choice kernels.
    2) Internally induced fluctuations.
    This measures fluctuations of reconstructed power values after the
    effect of contrast is subtracted out.
    3) A mixture of both.
    This uses the overall power fluctuations and correlates them with
    the choice kernel.
    """
    k, c = get_kernel(subject)
    res = {"subject": subject, "cluster": cluster, "latency": latency}
    try:
        ck_int = get_cortical_kernel(
            subject,
            hemi,
            cluster,
            latency,
            freq_band=freq_band,
            ogl=ogl,
            remove_contrast_induced_flucts=True,
        )
        ck_all = get_cortical_kernel(
            subject,
            hemi,
            cluster,
            latency,
            freq_band=freq_band,
            ogl=ogl,
            remove_contrast_induced_flucts=False,
        )
        res["AF-CIFcorr"] = np.corrcoef(ck_int.ravel(), k.ravel())[0, 1]
        res["AFcorr"] = np.corrcoef(ck_all.ravel(), k.ravel())[0, 1]

    except RuntimeError:
        # Area missing
        pass

    if peakslope is not None:
        # vfcpeak = dp.extract_peak_slope(ssd)
        if cluster in peakslope:
            pvt = peakslope.loc[:, cluster]
            assert pvt.shape == (10, 15)
            res["DCDcorr"] = np.corrcoef(k.ravel(), pvt.loc[:, subject].values)[0, 1]
            res["DCD-AFcorr"] = np.corrcoef(ck_all.ravel(), pvt.loc[:, subject].values)[
                0, 1
            ]

    return res


@memory.cache()
def build_design_matrix(
    subject, cluster, latency, hemi="Averaged", zscore=True, freq_bands=None, ogl=False
):
    if not ogl:
        filenames = glob(join(inpath, "S%i_*_%s_agg.hdf" % (subject, "stimulus")))
        data = asr.delayed_agg(filenames, hemi=hemi, cluster=cluster)()
    else:
        filenames = glob(
            join(inpath, "ogl/slimdown/ogl_S%i_*_%s_agg.hdf" % (subject, "stimulus"))
        )
        data = asr.delayed_agg(
            filenames, hemi="Pair", cluster=["%s_LH" % cluster, "%s_RH" % cluster]
        )().copy()

        data = data.groupby(["hemi", "trial", "freq"]).mean()
        data.head()
        hemi = np.asarray(data.index.get_level_values("hemi")).astype("str")
        trial = np.asarray(data.index.get_level_values("trial")).astype(int)
        freq = np.asarray(data.index.get_level_values("freq")).astype(int)
        # cl = hemi.copy()
        cl = [cluster] * len(hemi)
        index = pd.MultiIndex.from_arrays(
            [hemi, cl, trial, freq], names=["hemi", "cluster", "trial", "freq"]
        )
        data.index = index
        data.head()

    trial_index = data.index.get_level_values("trial").unique()

    X, time_per_col = regress.prep_low_level_data(
        cluster, data, 0, latency, trial_index, freq_bands=freq_bands
    )
    print(X.shape)
    cols = X.columns.values
    index = X.index.values
    X = X.values
    meta = da.preprocessing.get_meta_for_subject(subject, "stimulus")
    meta.set_index("hash", inplace=True)
    meta = meta.loc[trial_index]
    if zscore:
        X = (X - X.mean(0)) / X.std(0)

    return X, time_per_col, cols, meta

def plotwerr(pivottable, *args, ax=None, label=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    N = pivottable.shape[0]
    x = pivottable.columns.values
    mean = pivottable.mean(0).values
    std = pivottable.std(0).values
    sem = std/(N**.5)
    ax.plot(x, mean, *args, label=label, **kwargs)
    if 'alpha' in kwargs:
        del kwargs['alpha']
    if 'color' in kwargs:
        color = kwargs['color']
        del kwargs['color']
        ax.fill_between(x, mean+sem, mean-sem, facecolor=color, edgecolor='none', alpha=0.5, **kwargs)    
    else:
        ax.fill_between(x, mean+sem, mean-sem, edgecolor='none', alpha=0.5, **kwargs) 

