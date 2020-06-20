"""
Analyze behavior of single trials
"""
import numpy as np
import os
import pandas as pd

from conf_analysis.behavior import metadata
from conf_analysis.meg import preprocessing
from glob import glob
from os.path import join
from pymeg import aggregate_sr as asr

from joblib import Memory
from os.path import join
from pymeg import aggregate_sr as asr


if "TMPDIR" in os.environ.keys():
    memory = Memory(cachedir=os.environ["PYMEG_CACHE_DIR"])
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


def get_trials(subject, epoch, cluster, hemi):
    filenames = glob(join(inpath, "S%i_*_%s_agg.hdf" % (subject, epoch)))
    meta = preprocessing.get_meta_for_subject(subject, "stimulus").set_index("hash")
    agg = asr.delayed_agg(filenames, hemi=hemi, cluster=cluster)()
    idx = pd.Series([subject] * len(agg), index=agg.index)
    idx.name = "subject"
    agg.set_index(idx, append=True, inplace=True)
    return meta, agg


def samplewise(agg, pl):
    res = []
    for (subject, cluster), d in agg.groupby(["subject", "cluster"]):
        latencies = pl.loc[(subject,), cluster].values + np.arange(0, 1, 0.1)
        res.append(d.loc[:, latencies])
    return pd.concat(res)


def featmat(agg):
    """
    """
    return pd.pivot_table(index="trial", columns="freq", data=agg)


def trials_by_column(data, col):
    """
    Takes data, df that contains all trials for one cluster,
    and sorts by series col.
    """
    # Make sure data has only trials as index
    names = data.index.names
    for name in names:
        if name == "trial":
            continue
        data.index = data.index.droplevel(name)

    svals = col.index
    data = data.loc[svals, :]
    assert all(data.index.values == svals.values)
    return data


def trials_by_index(data, col):
    # Make sure data has only trials as index
    names = data.index.names
    for name in names:
        if name == "trial":
            continue
        data.index = data.index.droplevel(name)

    data = data.loc[col, :]
    assert all(data.index.values == col.values)
    return data


def classifier_by_freq(agg, meta):
    from sklearn.linear_model import LogisticRegression

    coefs = []
    for f, df in agg.groupby("freq"):
        trials, freqs, samples, fm = featmat(df)
        clf_d = LogisticRegression(max_iter=20000, class_weight="balanced").fit(
            fm, meta.loc[trials].response.values
        )
        coefs.append(clf_d.coef_)
    return coefs


def avg_freqs(data, bins=np.array([0, 10, 25, 75, 150])):
    bins = np.asarray(bins)
    ctrs = (bins[1:] + bins[:-1]) / 2

    cutter = pd.cut(data.index.get_level_values("freq"), bins, labels=ctrs)
    cutter.name = "freq"
    dp = data.groupby([cutter, "subject", "cluster", "trial"]).mean()
    dp.index.names = ["freq", "subject", "cluster", "trial"]
    return dp


@memory.cache()
def all_auc_by_TFR(cluster, hemi, epoch):
    res = []
    for subject in range(1, 16):
        meta, agg = get_trials(subject, epoch, cluster, hemi)
        r = get_auc_by_TFR(agg, meta.response)
        r.set_index(agg.columns.get_level_values("time"), inplace=True)
        r.loc[:, "subject"] = subject
        res.append(r.set_index("subject", append=True))
    return pd.concat(res)


@memory.cache()
def get_auc_by_TFR(data, choices, bins=np.array([0, 10, 25, 75, 150])):
    ctrs = (bins[1:] + bins[:-1]) / 2
    cutter = pd.cut(data.index.get_level_values("freq"), bins, labels=ctrs)
    res = {}
    data = data.groupby([cutter, "trial"]).mean()

    for center in ctrs:
        df = data.loc[(center, slice(None)), :]
        idx = df.index.get_level_values("trial")
        c = choices.loc[idx].values
        res[center] = kernel(df.values, c)
    return pd.DataFrame(res)


def kernel(data, choices):
    """
    Compute ROC AUC for choice weights.
    """
    from sklearn.metrics import roc_auc_score

    kernel = [roc_auc_score(choices, column) for column in data.T]
    return kernel


def CRFS(pl, avg=False, stimulus=True):
    from itertools import product

    if stimulus:
        return pd.concat(
            [
                get_all_cia(pl, hemi=hemi, cluster=cluster, avg=avg)
                for cluster, hemi in product(
                    ["vfcPrimary", "JWG_M1", "vfcFEF"], ["Averaged", "Lateralized"]
                )
            ]
        )
    else:
        return pd.concat(
            [
                get_response_cia(pl, hemi=hemi, cluster=cluster)
                for cluster, hemi in product(
                    ["vfcPrimary", "JWG_M1", "vfcFEF"], ["Averaged", "Lateralized"]
                )
            ]
        )


def get_all_cia(pl, hemi="Averaged", cluster="vfcPrimary", avg=False, epoch="stimulus"):

    frames = []
    for subject in range(1, 16):
        meta, agg = get_trials(subject, epoch, cluster, hemi)
        agg = agg.groupby(["freq", "trial"]).mean()
        times = agg.columns.get_level_values("time")
        for sample in range(10):
            peak = pl.loc[(subject, sample), cluster]
            idt = times[np.argmin(np.abs(times - peak))]
            if avg:
                t = (
                    agg.loc[:, idt]
                    .groupby("freq")
                    .apply(
                        lambda x: contrast_integrated_averages(
                            x,
                            meta,
                            slice(0, sample + 1),
                            centers=np.linspace(0.2, 0.8, 31),
                            mean_sample=True,
                        )
                    )
                    .unstack()
                )
            else:
                t = (
                    agg.loc[:, idt]
                    .groupby("freq")
                    .apply(
                        lambda x: contrast_integrated_averages(
                            x, meta, sample, centers=np.linspace(0.2, 0.8, 31)
                        )
                    )
                    .unstack()
                )

            t.columns = t.columns.droplevel("time")
            t.loc[:, "subject"] = subject
            t.loc[:, "sample"] = sample
            t.loc[:, "cluster"] = cluster
            t.loc[:, "hemi"] = hemi
            t = t.reset_index().set_index(
                ["subject", "hemi", "cluster", "sample", "freq"]
            )
            frames.append(t)
    return pd.concat(frames)


def get_response_cia(pl, hemi="Averaged", cluster="vfcPrimary", epoch="response"):

    frames = []
    for subject in range(1, 16):
        meta, agg = get_trials(subject, epoch, cluster, hemi)
        agg = agg.groupby(["freq", "trial"]).mean()
        times = agg.columns.get_level_values("time")
        for latency in times[(-0.5 < times) & (times < 0.25)]:

            idt = times[np.argmin(np.abs(times - latency))]

            t = (
                agg.loc[:, idt]
                .groupby("freq")
                .apply(
                    lambda x: contrast_integrated_averages(
                        x,
                        meta,
                        slice(0, 11),
                        centers=np.linspace(0.2, 0.8, 31),
                        mean_sample=True,
                    )
                )
                .unstack()
            )

            t.columns = t.columns.droplevel("time")
            t.loc[:, "subject"] = subject
            t.loc[:, "latency"] = latency
            t.loc[:, "cluster"] = cluster
            t.loc[:, "hemi"] = hemi
            t = t.reset_index().set_index(
                ["subject", "hemi", "cluster", "latency", "freq"]
            )
            frames.append(t)
    return pd.concat(frames)


def contrast_integrated_averages(
    agg, meta, sample, centers=np.linspace(0.1, 0.9, 5), width=0.2, mean_sample=False
):
    """
    For each colum in agg compute contrast integrated average
    """
    agg = agg.groupby("trial").mean()

    contrast = np.stack(meta.loc[agg.index.values, "contrast_probe"])[:, sample]
    if mean_sample and (len(contrast.shape) > 1):
        s = contrast.shape
        contrast = contrast.mean(1)
        # print(s, '->', contrast.shape)
    w = width / 2.0
    rows = []
    for center in centers:
        idx = ((center - w) < contrast) & (contrast < (center + w))
        r = agg.loc[idx, :].mean()
        # r.loc['contrast'] = center
        # r.set_index('contrast', inplace=True, append=True)
        rows.append(r)
    rows = pd.concat(rows, 1).T
    rows = rows.set_index(centers)
    rows.index.name = "contrast"
    c50 = ((0.5 - w) < contrast) & (contrast < (0.5 + w))
    r = agg.loc[c50, :].mean().values
    return pd.DataFrame(rows.values - r, index=rows.index, columns=rows.columns)


def cia(power, contrast, centers=np.linspace(0.1, 0.9, 5), width=0.2):
    w = width / 2.0
    rows = []
    for center in centers:
        idx = ((center - w) < contrast) & (contrast < (center + w))
        r = power[idx].mean()
        rows.append(r)
    return centers, np.array(rows)


def plot_crfs(crfs, freqbins=[0, 10, 25, 45, 80, 150]):
    import seaborn as sns
    import pylab as plt

    freqbins = np.asarray(freqbins)
    freq = crfs.index.get_level_values("freq")
    centers = freqbins[0:-1] + np.diff(freqbins)
    cutter = pd.cut(freq, bins=freqbins, labels=centers)
    crfs = crfs.groupby([cutter, "subject", "contrast", "hemi", "cluster"]).mean()
    crfs.index.names = ["freq", "subject", "contrast", "hemi", "cluster"]
    g = sns.FacetGrid(data=crfs.reset_index(), rows="hemi", cols="cluster", hue="freq")
    g.map(plt.plot, "contrast", "power")


def std_corr(data, crfs):
    from conf_analysis.behavior import empirical
    from scipy.stats import linregress

    bias = data.groupby(["snum"]).apply(empirical.crit).reset_index()
    bias.columns = ["subject", "bias"]
    bias.set_index("subject", inplace=True)
    dp = data.groupby(["snum"]).apply(empirical.dp).reset_index()
    dp.columns = ["subject", "dp"]
    dp.set_index("subject", inplace=True)

    # crfs = crfs.groupby(['subject', 'freq', 'contrast']).mean()

    crfsd = (
        crfs.groupby(["subject", "freq"]).max()
        - crfs.groupby(["subject", "freq"]).min()
    )
    crfsd.name = "power"
    crfsd = pd.pivot_table(
        crfsd.reset_index(), columns="freq", index="subject", values="power"
    )

    # crfs = crfs.groupby(['subject', 'bfreq']).mean()
    contrast = crfs.index.get_level_values("contrast")

    crfs = (
        crfs.loc[(0.6 < contrast) & (contrast < 0.7)]
        .groupby(["subject", "freq"])
        .mean()
    ) - (
        np.abs(
            crfs.loc[(0.3 < contrast) & (contrast < 0.4)]
            .groupby(["subject", "freq"])
            .mean()
        )
    )
    crfs.name = "power"
    crfs = pd.pivot_table(
        crfs.reset_index(), columns="freq", index="subject", values="power"
    )

    results = []
    for freq in crfs:
        df = crfs.loc[:, freq]
        dff = crfsd.loc[:, freq]
        slope, _, d, p_d, _ = linregress(dp.loc[df.index, "dp"].values, dff.values)
        slope, _, c, p_c, _ = linregress(bias.loc[df.index, "bias"], df)
        results.append(
            {"freq": freq, "dp": d, "bias": c, "pval_dp": p_d, "pval_bias": p_c}
        )
    return pd.DataFrame(results)
