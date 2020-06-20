import numpy as np
import os
import pandas as pd
import pylab as plt
import seaborn as sns

import matplotlib
from matplotlib import cm
from matplotlib import colors
from matplotlib import gridspec
from matplotlib import pyplot
from pymeg import atlas_glasser as ag
from pymeg import source_reconstruction as sr

from conf_analysis.behavior import metadata
from pymeg.contrast_tfr_plots import PlotConfig
from joblib import Memory

import pickle

if "RRZ_LOCAL_TMPDIR" in os.environ.keys():
    memory = Memory(location=os.environ["RRZ_LOCAL_TMPDIR"], verbose=-1)
if "TMPDIR" in os.environ.keys():
    tmpdir = os.environ["TMPDIR"]
    memory = Memory(location=tmpdir, verbose=-1)
else:
    memory = Memory(location=metadata.cachedir, verbose=-1)


choice_decoding_areas = (
    "HCPMMP1_premotor",
    "JWG_M1",
    "vfcFEF",
    "JWG_IPS_PCeS",
    "HCPMMP1_dlpfc",
    "JWG_aIPS",
)


plot_config = PlotConfig(
    {"stimulus": (-0.35, 1.1), "response": (-0.35, 0.1)},  # Time windows for epochs
    ["CONF_signed", "CONF_unsigned", "MIDC_split"],  # Signal names
)

plot_config.configure_epoch(
    "stimulus",
    **{
        "xticks": [0,1],
        "xticklabels": ['0\n Stim.     \non ', '1\nStim.    \noff   '],            
        "yticks": [0.25, 0.5, 0.75],
        "yticklabels": [0.25, 0.50, 0.75],
        "xlabel": "",
        "ylabel": "AUC",
    },
)
plot_config.configure_epoch(
    "response",
    **{
        "xticks": [0],
        "xticklabels": ['0\n       Resp-\n      onse'],
        "yticks": [0.25, 0.5, 0.75],
        "yticklabels": [0.25, 0.50, 0.75],
        "xlabel": "",
        "ylabel": "AUC",
    },
)


def filter_latency(data, min, max):
    lat = data.index.get_level_values("latency").values
    return data.loc[(min < lat) & (lat < max), :]


def get_decoding_data(decoding_classifier="SCVlin", restrict=True, ogl=False):   
    def _get_decoding_data(file, restrict): 
        df = pd.read_hdf(file)
        df.loc[:, "latency"] = df.latency.round(3)
        idnan = np.isnan(df.subject)
        df.loc[idnan, "subject"] = df.loc[idnan, "sub"]
        df = df.loc[~np.isnan(df.subject), :]
        df = df.query('Classifier=="%s"' % decoding_classifier)
        df.loc[:, "cluster"] = [
            (c.split(" ")[0].replace("_LH", "").replace("_RH", ""))
            for c in df.loc[:, "cluster"].values
        ]
        if restrict:
            clusters = ag.areas.values()
            idx = [True if c in clusters else False for c in df.loc[:, "cluster"]]
            df = df.loc[idx, :]
        #for field in ["hemi", "cluster", "Classifier", "epoch"]:
        #    df.loc[:, field] = df.loc[:, field].astype("category")
        if 'mc<0.5' in df.columns:
            df.loc[:, "split"] = df.loc[:, "mc<0.5"].astype(str)
            del df['mc<0.5']
        elif 'R1' in df.columns:
            df.loc[:, "split"] = df.loc[:, "R1"].astype(str)
            del df['R1']

        df.set_index(
            [
                "Classifier",
                "signal",
                "subject",
                "epoch",
                "latency",
                "split",
                "hemi",
                "cluster",
            ],
            inplace=True,
        )
        df = df.loc[~df.index.duplicated()]
        df = df.unstack(["hemi", "cluster"])
        if not ogl:
            idt = df.test_accuracy.index.get_level_values("signal") == "CONF_signed"
            df.loc[idt, "test_accuracy"] = (df.loc[idt, "test_accuracy"] - 0.25).values
            df.loc[~idt, "test_accuracy"] = (df.loc[~idt, "test_accuracy"] - 0.5).values
        return df


    if not ogl:        
        df = _get_decoding_data("/Users/nwilming/u/conf_analysis/results/all_decoding_20190513.hdf", restrict)
        df2 = _get_decoding_data("/Users/nwilming/u/conf_analysis/results/all_decoding_conf_resp_split.hdf", restrict)
        df = pd.concat([df, df2])#, ignore_index=True).set_index(["signal", "subject", "epoch","latency", "split","hemi","cluster"])
    else:        
        df = _get_decoding_data("/Users/nwilming/u/conf_analysis/results/all_decoding_ogl_merged_20190527.hdf", restrict)
    #df.loc[:, 'signal'] = df.loc[:, 'signal'].astype("category")
    return df
    



def get_ssd_data(ssd_classifier="Ridge", restrict=True, ogl=False):


    if not ogl:
        df = pd.read_hdf(
            "/Users/nwilming/u/conf_analysis/results/all_decoding_SSD_20190513.hdf"
        )
    else:
        df = pd.read_hdf(
            "/Users/nwilming/u/conf_analysis/results/all_decoding_ogl_ssd_merged_20190527.hdf"
        )

    df = df.loc[~np.isnan(df.subject), :]
    df = df.query('Classifier=="%s"' % ssd_classifier)
    df.loc[:, "cluster"] = [
        c.split(" ")[0].replace("_LH", "").replace("_RH", "")
        for c in df.loc[:, "cluster"].values
    ]
    if restrict:
        clusters = ag.areas.values()
        idx = [True if c in clusters else False for c in df.loc[:, "cluster"]]
        df = df.loc[idx, :]
    for field in ["signal", "hemi", "cluster", "Classifier", "epoch"]:
        df.loc[:, field] = df.loc[:, field].astype("category")
    df.set_index(
        [
            "Classifier",
            "signal",
            "subject",
            "epoch",
            "latency",
            "sample",
            "hemi",
            "cluster",
        ],
        inplace=True,
    )
    df = df.loc[~df.index.duplicated()]
    df = df.unstack(["hemi", "cluster"])
    # Round latency to 3 digits to make them comparable
    o = df.index.to_frame()
    o.loc[:, "latency"] = o.latency.round(3)
    df.index = pd.MultiIndex.from_frame(o)
    return df



def get_ssd_var_data(ssd_classifier="Ridge", restrict=True):
    df = pd.read_hdf(
            "/Users/nwilming/u/conf_analysis/results/ssd_decoding_by_var_extra_analysis_20191203.hdf"
        )
    

    df = df.loc[~np.isnan(df.subject), :]
    df = df.query('Classifier=="%s"' % ssd_classifier)
    df.loc[:, "cluster"] = [
        c.split(" ")[0].replace("_LH", "").replace("_RH", "")
        for c in df.loc[:, "cluster"].values
    ]
    if restrict:
        clusters = ag.areas.values()
        idx = [True if c in clusters else False for c in df.loc[:, "cluster"]]
        df = df.loc[idx, :]
    for field in ["signal", "hemi", "cluster", "Classifier", "epoch"]:
        df.loc[:, field] = df.loc[:, field].astype("category")
    df.set_index(
        [
            "Classifier",
            "signal",
            "subject",
            "epoch",
            "latency",
            "sample",
            "hemi",
            "cluster",
            "variance",
        ],
        inplace=True,
    )
    df = df.loc[~df.index.duplicated()]
    df = df.unstack(["hemi", "cluster"])
    # Round latency to 3 digits to make them comparable
    o = df.index.to_frame()
    o.loc[:, "latency"] = o.latency.round(3)
    df.index = pd.MultiIndex.from_frame(o)
    return df


split_label = "split"


@memory.cache()
def _get_data(df, signal, hemi, cluster, epoch, split=None):
    dclust = (
        df.query('signal=="%s" & epoch=="%s"' % (signal, epoch))
        .loc[:, cluster]
        .to_frame()
    )
    dbase = (
        df.query('signal=="%s" & epoch=="stimulus"' % signal).loc[:, cluster].to_frame()
    )
    if not split:
        values = pd.pivot_table(
            data=dclust, index="subject", columns="latency", values=cluster
        )
        base = pd.pivot_table(
            data=dbase, index="subject", columns="latency", values=cluster
        )
        return {"nosplit": (values, base)}
    else:
        res = {}
        for split, ds in dclust.groupby(split_label):
            values = pd.pivot_table(
                data=ds, index="subject", columns="latency", values=cluster
            )
            idx = dbase.index.get_level_values(split_label) == split
            base = pd.pivot_table(
                data=dbase.loc[idx], index="subject", columns="latency", values=cluster
            )
            if split == "True":
                base = -base + 1
                values = -values + 1
            res[split] = values, base
        return res


class StreamPlotter(object):
    def __init__(self, conf, signals=None, datasets=None, gs=None, title_palette={}):
        from collections import namedtuple
        self.super_gs=gs
        self.title_palette=title_palette
        Plot = namedtuple("Plot", ["name", "cluster", "location", "annot_y", "annot_x"])
        top, middle, bottom = slice(0, 2), slice(1, 3), slice(2, 4)
        # fmt: off
        self.layout = [
            Plot("V1", "vfcPrimary", [0, middle], True, True),
            Plot("V2-V4", "vfcEarly", [1, middle], False, False),
            # Dorsal
            Plot("V3A/B", "vfcV3ab", [2, top], False, False),
            Plot("IPS0/1", "vfcIPS01", [3, top], False, False),
            Plot("IPS2/3", "vfcIPS23", [4, top], False, False),
            Plot("aIPS", "JWG_aIPS", [5, top], False, False),
            
            # Ventral
            Plot("LO1/2", "vfcLO", [2, bottom], False, False),
            Plot("MT/MST", "vfcTO", [3, bottom], False, False),
            Plot("VO1/2", "vfcVO", [4, bottom], False, False),
            Plot("PHC", "vfcPHC", [5, bottom], False, False),
            
            Plot("IPS/PostCeS", "JWG_IPS_PCeS", [6, middle], False, False),
            Plot("M1-hand", "JWG_M1", [7, middle], False, False),
        ]
        # fmt: on
        self.configuration = conf
        self.ax = {}
        self.data = {}
        if signals is not None:
            self.add_signals(signals)
        if datasets is not None:
            for key, value in datasets.items():
                self.add_dataset(key, value)
        self.ratio = (
            np.diff(conf.time_windows["response"])[0]
            / np.diff(conf.time_windows["stimulus"])[0]
        )
        self.cache = (
            "/Users/nwilming/u/conf_analysis/results/pymc_fig_2_posterior.pickle"
        )
        try:
            print("Now Loading posterioe")
            self.post = pickle.load(open(self.cache, "rb"))
        except IOError:
            print("No posterior loaded")
            self.post = {}
        self.cluster_test_cache = (
            "/Users/nwilming/u/conf_analysis/results/pymc_fig_2_cluster.pickle"
        )
        try:
            self.cluster = pickle.load(open(self.cluster_test_cache, "rb"))
        except IOError:
            print("No cluster test loaded")
            self.cluster = {}

    def add_dataset(self, key, df):
        """
        Datasets will be plotted in different subplots
        """
        self.data[key] = df

    def add_signals(self, signals):
        """
        Different signal will be plotted in the same subplot
        """
        self.signals = signals

    def get_data(self, signal, hemi, cluster, epoch, split=None):
        df = self.data[hemi]
        return _get_data(df, signal, hemi, cluster, epoch, split)

    def get_posterior(self, signal, hemi, cluster, epoch):
        import pymc3 as pm

        try:
            return self.post[(signal, hemi, cluster, epoch)]
        except KeyError:
            pass
        print("Get posterior:", signal, hemi, cluster, epoch)
        data = self.get_data(signal, hemi, cluster, epoch, "split" in signal)
        results = {}
        for split, (values, base) in data.items():
            _, hdi, phdi, samples, model = get_baseline_posterior(
                base.loc[:, -0.25:0], values
            )
            mu = samples.get_values("mu_ind")
            ppc = pm.sample_posterior_predictive(samples, samples=2500, model=model)
            phdi = pm.stats.hpd(ppc["Out"].mean(2).ravel(), 0.01)
            hdi = pm.stats.hpd(mu, 0.01)
            results[split] = (values.columns.values, values.mean(0), hdi, phdi)

        self.post[(signal, hemi, cluster, epoch)] = results
        print("Dumping pickle: ", len(results))
        pickle.dump(self.post, open(self.cache, "wb"))
        return results

    def get_cluster_result(self, signal, hemi, cluster, epoch):
        try:
            return self.cluster[(signal, hemi, cluster, epoch)]
        except KeyError:
            pass
        print("Get cluster:", signal, hemi, cluster, epoch)
        from mne.stats import permutation_cluster_1samp_test as cluster_test
        from joblib import parallel_backend

        data = self.get_data(signal, hemi, cluster, epoch, "split" in signal)
        results = {}
        with parallel_backend("multiprocessing", n_jobs=12):
            for split, (values, base) in data.items():
                t_obs, _, p, H0 = cluster_test(
                    values.values - 0.5,
                    threshold={"start": 0, "step": 0.2},
                    connectivity=None,
                    tail=0,
                    n_permutations=1000,
                    n_jobs=12,
                )
                results[split] = (t_obs, p)
        self.cluster[(signal, hemi, cluster, epoch)] = results
        print("Dumping cluster pickle")
        pickle.dump(self.cluster, open(self.cluster_test_cache, "wb"))
        return results

    def plot(self, stats=False, flip_cbar=False, palette=None, aspect='auto'):
        if self.super_gs is None:
            self.gs = matplotlib.gridspec.GridSpec(
                len(self.data), 1, #width_ratios=[0.99, 0.01]
            )
        else:
            print('Making nested subplot')
            self.gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
                len(self.data), 1, #width_ratios=[0.99, 0.01],
                subplot_spec=self.super_gs,
            )


        for i, (name, dataset) in enumerate(self.data.items()):

            nr_cols = np.max([p.location[0] for p in self.layout]) + 1
            wr = list(np.tile([1, self.ratio, 0.1], nr_cols))
            print(wr)
            subgs = matplotlib.gridspec.GridSpecFromSubplotSpec(
                4,
                (nr_cols * 3),
                width_ratios=wr,
                subplot_spec=self.gs[i, 0],
                wspace=0.3,
                hspace=0.5,
            )

            for signal, color in self.signals.items():
                print(signal, color)
                self.plot_decoding_selected_rois(
                    signal,
                    name,
                    self.layout,
                    subgs,
                    conf=self.configuration,
                    color=color,
                    ax_prefix=i,
                    aspect=aspect,
                )

    def plot_decoding_selected_rois(
        self,
        signal,
        hemi,
        layout,
        gs,
        conf=None,
        palette=None,
        color=None,
        ax_prefix=None,
        aspect='auto'
    ):
        if palette is None:
            palette = get_area_palette()
        if color is not None:
            palette = {P.cluster: color for P in layout}

        fig = None
        print(palette)
        for P in layout:
            cluster = P.cluster

            for j, timelock in enumerate(["stimulus", "response"]):
                cluster_name = P.name
                # tfr to plot:
                col, row = P.location
                col = col * 3 + j
                ax_key = "%s_%s_%s_%s_%s" % (ax_prefix, j, col, row, cluster)
                try:
                    ax = self.ax[ax_key]
                except KeyError:
                    ax = plt.subplot(gs[row, col])
                    self.ax[ax_key] = ax

                time_cutoff = conf.time_windows[timelock]  # (-0.2, 1.1)

                self.plot_one_signal(
                    signal,
                    hemi,
                    cluster,
                    timelock,
                    palette[cluster],
                    "test_roc_auc",
                    plot_uncertainty=True,
                    ax=ax,
                )
                ax.set_aspect(aspect)
                # print(timelock, ax.get_xlim(), end='-->')
                conf.markup(timelock, ax, left=P.annot_y, bottom=P.annot_x)
                ax.set_xlim(conf.time_windows[timelock])
                ax.axvline(0, color="k",  ls="--", lw=0.75, zorder=-10, alpha=0.5)
                if timelock=='stimulus':
                    ax.axvline(1, color="k",   ls="--", lw=0.75, zorder=-10 , alpha=0.5)
                # print(ax.get_xlim())
                ax.set_ylim(0.15, 0.85)
                if j == 1:
                    ax.set_yticks([])
                    ax.set_yticklabels([])
                    ax.set_ylabel("")
                    ax.set_xlabel("")
                if j == 0:
                    if cluster in self.title_palette:
                        plt.title(cluster_name, 
                            {"fontsize": 7, 'color':self.title_palette[cluster]},
                            y=0.93, x=1.015,)
                    else:
                        plt.title(cluster_name, {"fontsize": 7})
                else:
                    plt.title("", {"fontsize": 8})

                sns.despine(ax=ax)
        return gs

    def plot_one_signal(
        self,
        signal,
        hemi,
        cluster,
        epoch,
        color,
        measure,
        plot_uncertainty=True,
        ax=None,
        **kw,
    ):
        """
        Plot one signal for one cluster. 

        Input data is a data frame with one column that encodes the cluster.
        """
        if ax is None:
            ax = plt.gca()
        try:
            data = self.get_posterior(signal, hemi, cluster, epoch)
        except KeyError:
            return
        cluster_test = self.get_cluster_result(signal, hemi, cluster, epoch)
        for key, (latency, mu, hdi, phdi) in data.items():
            ax.plot(latency, latency * 0 + 0.5, "k", zorder=-1)

            i = 0
            col = color[0].lower()
            offsets = {"k":0.2, "r": 0.2, "b": 0.19, "g": 0.18}

            ax.fill_between(
                latency,
                hdi[:, 0],
                hdi[:, 1],
                facecolor=col,
                alpha=0.25,
                edgecolor=(0, 0, 0, 0),
                lw=0,
            )
            tvals, pvals = cluster_test[key]
            if key == "False":
                ax.plot(latency, mu, color=col)
                draw_sig(ax, latency, pvals < 0.05, offsets[col]+0.015, (0.5,0.5,0.5))
            else:
                ax.plot(latency, mu, ':', color=col)
                draw_sig(ax, latency, pvals < 0.05, offsets[col], (0.5,0.5,0.5), linestyle='-', zorder=-1)
                draw_sig(ax, latency, pvals < 0.05, offsets[col], (0.25,0.25,0.25), linestyle=':')
            
            if sum(pvals<0.05)>0:
                lat = latency[pvals<0.05].min()
                print('DCD, %s, %s, earliest significant: %f'%(cluster, signal, lat))            
            #else:
            #    print('DCD, %s, %s, earliest significant: None'%(cluster, signal))            
            


def draw_sig(ax, x, id_sig, offset, color, linestyle='-', zorder=0):
    dt = np.diff(x)[0]
    # print(dt)
    id_sig = np.array([0] + list(id_sig.astype(float)) + [0])
    x = np.array(list(x) + [x[-1]])
    d = np.where(np.diff(id_sig) != 0)[0]
    # print(d)
    for low, high in zip(d[0::2], d[1::2]):
        ax.plot([x[low], x[high]], [offset, offset], linestyle=linestyle, color=color, lw=0.5, zorder=zorder)


def multicolor_line(ax, x, y, colorval, cmin, cmax, cmap="Reds"):
    # https://matplotlib.org/gallery/lines_bars_and_markers/multicolored_line.html
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm

    xnew = np.linspace(x.min(), x.max(), len(x) * 5)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(cmin, cmax)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(colorval)
    return ax.add_collection(lc)


def get_cvals(epoch, data):
    if epoch == "response":
        latency = 0
    else:
        latency = 1.25
    p = data.query("latency==%f" % latency).groupby("latency").mean().max()
    return {k: v for k, v in p.items()}


@memory.cache
def get_posterior(data):
    """
    Get uncertainty around average mean by means of bayesian 
    inference. For AUC values, nothing else.
    """
    import pymc3 as pm

    n_t = data.shape[1]
    with pm.Model():
        mu = pm.Normal("mu", 0.5, 1, shape=n_t)
        std = pm.Uniform("std", lower=0, upper=5, shape=n_t)
        v = pm.Exponential("ν_minus_one", 1 / 29.0) + 1
        pm.StudentT("Pred", mu=mu, lam=std ** -2, nu=v, shape=n_t)
        pm.StudentT("Out", mu=mu, lam=std ** -2, nu=v, shape=n_t, observed=data)

        k = pm.sample()
    mu = k.get_values("mu")
    samps = k.get_values("Pred")
    return mu.mean(0), pm.stats.hpd(mu), samps


@memory.cache
def get_baseline_posterior(data, all_data):
    """
    Get uncertainty around average mean by means of bayesian 
    inference. For AUC values, nothing else.

    data is n_time x n_obs
    """
    import pymc3 as pm

    n_t = data.shape[0]
    n_all = all_data.shape[1]
    with pm.Model() as model:
        mu = pm.Normal("mu", 0.5, 1)

        std_t = pm.Uniform("std_t", lower=0, upper=5, shape=n_t)
        mu_t = pm.Normal("mu_t", mu, std_t, shape=n_t)
        v = pm.Exponential("ν_minus_one", 1 / 29.0) + 1

        pm.StudentT("Out", mu=mu_t, lam=std_t ** -2, nu=v, shape=n_t, observed=data.T)

        mu_ind = pm.Normal("mu_ind", 0.5, 1, shape=n_all)
        std_ind = pm.Uniform("std_ind", lower=0, upper=5, shape=n_all)
        pm.StudentT(
            "Out_ind",
            mu=mu_ind,
            lam=std_ind ** -2,
            nu=v,
            shape=n_all,
            observed=all_data,
        )

        k = pm.sample(tune=1500, draws=1500)
    mu = k.get_values("mu_ind")
    ppc = pm.sample_posterior_predictive(k, samples=5000, model=model)
    predictive_hdi = pm.stats.hpd(ppc["Out"].mean(2).ravel())
    return mu.mean(0), pm.stats.hpd(mu), predictive_hdi, k, model


def plot_individual_areas_with_stats(data, type="Pair"):
    palette = get_area_palette()
    for cluster in data.columns:
        plot_signals_hand(
            data.loc[:, cluster].reset_index(),
            palette,
            "AUC",
            midc_ylim=(0.1, 0.9),
            conf_ylim=(0.5, 0.7),
            cortex_cmap="RdBu_r",
            midc_ylim_cortex=(0.1, 0.9),
            conf_ylim_cortex=(0.3, 0.7),
            plot_uncertainty=True,
            suffix="_%s_%s" % (type, cluster),
            lw=1,
        )


def plot_signals_hand(
    data,
    palette,
    measure,
    classifier="svm",
    midc_ylim=(-0.25, 0.25),
    conf_ylim=(-0.05, 0.25),
    midc_ylim_cortex=(-0.25, 0.25),
    conf_ylim_cortex=(-0.05, 0.25),
    cortex_cmap="RdBu_r",
    suffix="all",
    plot_uncertainty=False,
    **kw,
):
    """
    Plot decoding signals.
    This is the place to start!
    """

    allc, vfc, glasser, jwg = ag.get_clusters()
    col_order = list(glasser.keys()) + list(vfc.keys()) + list(jwg.keys())
    plt.figure(figsize=(8, 6))
    combinations = [
        ("stimulus", "MIDC_nosplit"),
        ("stimulus", "MIDC_split"),
        ("response", "MIDC_split"),
        (None, None),
        ("stimulus", "CONF_signed"),
        ("response", "CONF_signed"),
        (None, None),
        ("stimulus", "CONF_unsigned"),
        ("response", "CONF_unsigned"),
    ]
    index = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    gs = gridspec.GridSpec(3, 6)

    for i, (epoch, signal) in enumerate(combinations):
        if epoch is None:
            continue
        row, col = index[i]
        plt.subplot(gs[row, col * 2])
        d = data.query('epoch=="%s" & signal=="%s"' % (epoch, signal))

        cvals = {}
        d = d.groupby(["subject", "latency", split_label]).mean()
        for split, ds in d.groupby(split_label):
            for column in col_order:
                try:
                    values = pd.pivot_table(
                        data=ds, index="subject", columns="latency", values=column
                    )
                    latency = values.columns.values
                    if split == "True":
                        values = -values + 1

                    if plot_uncertainty:
                        mu, hdi = get_posterior(values.values)
                        plt.plot(latency, latency * 0 + 0.5, "k", zorder=-1)
                        plt.plot(latency, mu, color="k", **kw)
                        plt.fill_between(
                            latency,
                            hdi[:, 0],
                            hdi[:, 1],
                            color=palette[column],
                            alpha=0.75,
                        )
                    else:
                        plt.plot(
                            latency, values.values.mean(0), color=palette[column], **kw
                        )
                except KeyError:
                    pass

        cylim = []
        if "MIDC" in signal:
            plt.ylim(midc_ylim)
            cylim = midc_ylim_cortex
            # pyplot.locator_params(nticks=5)
        else:
            plt.ylim(conf_ylim)
            cylim = midc_ylim_cortex
            # pyplot.locator_params(nticks=5)

        if (col == 0) or ((col == 1) and (row > 0)):
            plt.ylabel("%s\n\n" % signal + r"$%s$" % measure)
        if (row == 2) or ((row == 0) and (col == 0)):
            plt.xlabel(r"$time$")
        center = (plt.xlim()[1] + plt.xlim()[0]) / 2.0
        # plt.text(plt.xlim()[0] + 0.05, 0.18, signal, size=8)

        sns.despine(ax=plt.gca(), trim=False)
        if epoch == "response":
            plt.axvline(0, color="k", zorder=-1, alpha=0.9)
        else:
            plt.axvline(1.25, color="k", zorder=-1, alpha=0.9)
        vmin, vmax = cylim
        plt.subplot(gs[row, 1 + (col * 2)])
        cvals = get_cvals(epoch, d)
        k = get_pycortex_plot(
            cvals, "fsaverage", vmin=vmin, vmax=vmax, cmap=cortex_cmap
        )
        ax = plt.imshow(k)
        plt.xticks([])
        plt.yticks([])
        sns.despine(ax=plt.gca(), left=True, bottom=True, right=True, top=True)
    plt.savefig(
        "/Users/nwilming/Dropbox/UKE/confidence_study/all_signals_%s.svg" % suffix
    )
    plt.savefig(
        "/Users/nwilming/Dropbox/UKE/confidence_study/all_signals_%s.pdf" % suffix
    )


def table_performance(
    df,
    t_stim=1.1,
    t_resp=0,
    areas=None,
    sortby="MIDC_split",
    signals=[
        "MIDC_split",        
    ],
    ignore_resp=True
):
    """
    Output a table of decoding performances, sorted by one signal.
    """
    df = df.query('signal=="MIDC_split"')
    if not ignore_resp:
        df_response = (
            df.query("epoch=='response' & (latency==%f)" % t_resp)
            .groupby("signal")
            .mean()
            .T.loc[:, signals]
        )
        df_response = df_response.sort_values(by=sortby, ascending=False)
        df_response.columns = pd.MultiIndex.from_tuples(
            [("response", x) for x in df_response.columns.values],
            names=["Epoch", "Cluster"],
        )
        df_stim = (
            df.query("epoch=='stimulus' & (latency==%f)" % t_stim)
            .groupby("signal")
            .mean()
            .T.loc[df_response.index, signals]
        )
    else:
        df_stim = (
            df.query("epoch=='stimulus' & (latency==%f)" % t_stim)
            .groupby("signal")
            .mean()
            .T.loc[:, signals]
        )
        df_stim.loc[:, 'STD'] = (
            df.query("epoch=='stimulus' & (latency==%f)" % t_stim)
            .groupby("signal")
            .std()
            .T.loc[:, signals]/(15**.5)
        )
        df_stim = df_stim.sort_values(by=sortby, ascending=False)
    df_stim.columns = pd.MultiIndex.from_tuples(
        [("stimulus", x) for x in df_stim.columns.values], names=["Epoch", "Cluster"]
    )
    if not ignore_resp:
        return pd.concat([df_response, df_stim], 1).round(2)
    else:
        return df_stim.round(2)


def make_state_space_triplet(df):
    plt.subplot(2, 2, 1)
    state_space_plot(
        df.test_roc_auc.Lateralized,
        "MIDC_split",
        "CONF_unsigned",
        df_b=df.test_roc_auc.Lateralized,
    )
    plt.plot([0.5, 1], [0.5, 1], "k--", alpha=0.9, lw=1)
    plt.xlim([0.4, 0.9])
    plt.ylim([0.45, 0.7])
    plt.axhline(0.5, color="k", lw=1)
    plt.axvline(0.5, color="k", lw=1)
    plt.xlabel("Lateralized MIDC split")
    plt.ylabel("Lateralized CONF_unsigned")

    plt.subplot(2, 2, 2)
    state_space_plot(
        df.test_roc_auc.Lateralized,
        "MIDC_split",
        "CONF_unsigned",
        df_b=df.test_roc_auc.Averaged,
    )
    plt.plot([0.5, 1], [0.5, 1], "k--", alpha=0.9, lw=1)
    plt.xlim([0.4, 0.9])
    plt.ylim([0.45, 0.7])
    plt.axhline(0.5, color="k", lw=1)
    plt.axvline(0.5, color="k", lw=1)
    plt.xlabel("Lateralized MIDC split")
    plt.ylabel("Averaged CONF_unsigned")
    plt.subplot(2, 2, 4)
    state_space_plot(
        df.test_roc_auc.Averaged,
        "MIDC_split",
        "CONF_unsigned",
        df_b=df.test_roc_auc.Averaged,
    )
    plt.plot([0.5, 1], [0.5, 1], "k--", alpha=0.9, lw=1)
    plt.xlim([0.4, 0.9])
    plt.ylim([0.45, 0.7])
    plt.axhline(0.5, color="k", lw=1)
    plt.axvline(0.5, color="k", lw=1)
    plt.xlabel("Averaged MIDC split")
    plt.ylabel("Averaged CONF_unsigned")
    sns.despine(offset=5)
    plt.savefig("/Users/nwilming/Dropbox/UKE/confidence_study/state_space_plot.svg")
    plt.savefig("/Users/nwilming/Dropbox/UKE/confidence_study/state_space_plot.pdf")


def state_space_plot(df_a, signal_a, signal_b, df_b=None, a_max=0.6, color=None):
    def reshuffle(x, y):
        """Reshape the line represented by "x" and "y" into an array of individual
        segments.
        See: https://stackoverflow.com/questions/13622909/matplotlib-how-to-colorize-a-large-number-of-line-segments-as-independent-gradi/13649811#13649811
        """
        points = np.vstack([x, y]).T.reshape(-1, 1, 2)
        points = np.concatenate([points[:-1], points[1:]], axis=1)
        return points

    def plot_with_color(x, y, color, ax):
        from matplotlib.collections import LineCollection
        from matplotlib.colors import LinearSegmentedColormap

        palette = LinearSegmentedColormap.from_list("hrmpfh", [(1, 1, 1), color])
        segments = reshuffle(x, y)
        coll = LineCollection(segments, cmap=palette)
        coll.set_array(np.linspace(0, 1, len(x)))
        coll.set_linewidth(1)
        ax.add_collection(coll)

    palette = get_area_palette()
    if color is not None:
        palette = {k: color for k in palette.keys()}
    ax = plt.gca()
    if df_b is None:
        df_b = df_a
    for area in df_a.columns.get_level_values("cluster"):
        Aarea = pd.pivot_table(
            data=df_a.groupby(["signal", "latency"]).mean(),
            index="signal",
            columns="latency",
            values=area,
        )
        Barea = pd.pivot_table(
            data=df_b.groupby(["signal", "latency"]).mean(),
            index="signal",
            columns="latency",
            values=area,
        )
        try:
            color = palette[area]
        except KeyError:
            if color is None:
                color = "k"

        if Aarea.loc[signal_a, :].max() >= a_max:
            ax.plot(
                Aarea.loc[signal_a, :].values,
                Barea.loc[signal_b, :].values,
                color=color,
                alpha=0.5,
                label=area,
            )


"""
The next section plots decoding values from all signals and many areas at one time point.
"""


def get_signal_comp_data(df, latency, epoch="stimulus"):
    df_interest = df.query("epoch=='%s'" % epoch).query("latency==%f" % latency)
    # print(df_interest.index.get_level_values('epoch').unique())
    # df_base = df.query("epoch=='stimulus'").query("-0.25<latency<0")
    df_interest = df_interest.stack().reset_index()
    data = []
    signals = []
    subjects = []
    areas = []
    for signal in ["MIDC_split", "CONF_unsigned", "CONF_signed"]:
        X = pd.pivot_table(
            data=df_interest.query('signal=="%s"' % signal),
            index="cluster",
            columns="subject",
            values=0,
        )
        signals += [signal]
        areas = X.index.values.to_dense()
        subjects = X.columns.values
        data.append(X.values)
    data = np.stack(data)
    return np.array(signals), subjects, areas, data


def _plot_signal_comp(
    df, latency, idx, colors, title, epoch="stimulus", 
    plot_labels=False, alpha=0.05, auc_cutoff=0.6, ax=None,
    horizontal=False,
    color_by_cmap=False,
):
    from conf_analysis.meg import stats
    import pymc3 as pm
    if ax is None:
        ax=plt.gca()
    signals, subjects, areas, data = get_signal_comp_data(df, latency, epoch)
    signal_delta = {"MIDC_split": 0.15, "CONF_unsigned": 0, "CONF_signed": -0.15}
    id_midc = np.where(signals == "MIDC_split")[0]
    k, mdl = stats.auc_get_sig_cluster_posterior(data)
    mu = k.get_values("mu_ind")
    hdi = pm.stats.hpd(mu, alpha)
    # HDI = clusters x signals x hdi
    if idx is None:
        mean_data = data[id_midc, :].mean(-1).ravel()
        idx = np.argsort(mean_data).squeeze()
        idx = [i for i in idx if mean_data[i] > auc_cutoff]
    y = np.arange(len(idx))
    for isig, name in enumerate(signals):
        # Plot means
        x = data.mean(2)[isig, idx]

        # Plot hdi
        for il, icl in enumerate(idx):
            low, high = hdi[icl, isig]        
            xx, yy = x[il], y[il] + signal_delta[signals[isig]]
            if horizontal:
                yy, xx = xx, yy

            if color_by_cmap:
                color = colors[signals[isig]](yy)
            else:
                color = colors[signals[isig]]
            ax.plot(
                xx,
                yy,
                ".",
                markerfacecolor='k', markeredgecolor='none', alpha=0.25,
            )
            if not horizontal:                
                ax.plot(
                    [low, high],
                    np.array([il, il]) + signal_delta[signals[isig]],
                    color=color,
                )
            else:
                ax.plot(                    
                    np.array([il, il]) + signal_delta[signals[isig]],
                    [low, high],
                    color=color,  
                    zorder=-1
                )                
            if plot_labels and (name == "MIDC_split"):
                cluster = (
                    areas[idx[il]]
                    .replace("NSWFRONT_", "")
                    .replace("HCPMMP1_", "")
                    .replace("JWG_", "")
                )
                if not horizontal:
                    ax.text(1.05 * high, y[il], cluster)
                else:
                    ax.text(y[il], 1.05 * high, cluster)

    if not horizontal:
        plt.axvline(0.5, color="k", lw=0.5)
    else:
        plt.axhline(0.5, color="k", lw=0.5)
    # plt.xlim(xlim)
    plt.title(title)
    if not horizontal:
        plt.xlim([0.4, 0.75])
        plt.yticks([])
    else:
        plt.ylim([0.4, 0.75])
        plt.xticks([])
    sns.despine(left=True, ax=ax)
    return idx


def plot_signal_comp(
    df,
    latency=1.2,
    xlim=[0.49, 0.85],
    xlim_avg=[0.45, 0.6],
    auc_cutoff=0.6,
    gs=None,
    alpha=0.05,
    epoch="stimulus",
    idx=None,
    colors={"MIDC_split": "r", "CONF_unsigned": "b", "CONF_signed": "g"},
):
    """
    Call this function to plot all decoding values at latency=x in df
    during the response period.

    The plot will compare Averaged/Lateralized and MIDC_split with
    CONF_signed and CONF_unsigned.

    df should be output from get_decoding_data().test_roc_auc

    Statistics are done using Bayesian estimation and comparison
    against pre-stimulus baseline data (latency=0).

    For now only setting latency=0 works reliably with the stats
    (because latency = 0 -> pre stimulus baseline and interesting
    time point in response period.)
    """
    import joblib
    from conf_analysis.meg import stats
    import pymc3 as pm

    # Plot Lateralized first.
    if gs is None:
        gs = matplotlib.gridspec.GridSpec(1, 6)
    else:
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 6, gs)
    # plt.figure(figsize=(5.5, 10))
    ax = plt.subplot(gs[0, 4:])
    idx = _plot_signal_comp(
        df.test_roc_auc.Lateralized,
        latency,
        idx,
        colors,
        "Lateralized",
        epoch=epoch,
        plot_labels=True,
        alpha=alpha,
        auc_cutoff=0.6,
        ax=ax
    )
    plt.xlim(xlim)
    plt.xlabel("AUC")
    ax = plt.subplot(gs[0, :2])
    _plot_signal_comp(
        df.test_roc_auc.Averaged,
        latency,
        idx,
        colors,
        "Averaged",
        epoch=epoch,
        plot_labels=False,
        alpha=alpha,
        auc_cutoff=0.6,
        ax=ax
    )
    plt.xlim(xlim_avg[::-1])
    plt.xlabel("AUC")
    return idx, plt.subplot(gs[0, 2:4])


"""
The next section defines a color paletter and some plotting functions w/ pycortex.
"""


def get_area_palette(restrict=True):
    allc, vfc, glasser, jwg = ag.get_clusters()

    vfc_colors = sns.color_palette("Reds", n_colors=len(vfc) + 2)[1:-1]

    palette = {
        name: color
        for name, color in zip(list(vfc.keys()), vfc_colors)
        if name in ag.areas.values()
    }

    front_colors = sns.color_palette("Blues", n_colors=len(glasser) + 2)[1:-1]

    frontal = {
        name: color
        for name, color in zip(list(glasser.keys())[::-1], front_colors)
        if name in ag.areas.values()
    }

    jwdg_colors = sns.color_palette("Greens", n_colors=len(jwg) + 2)[1:-1]

    jwdg = {
        name: color
        for name, color in zip(list(jwg.keys())[::-1], jwdg_colors)
        if name in ag.areas.values()
    }
    palette.update(frontal)
    palette.update(jwdg)
    return palette


@memory.cache
def get_pycortex_plot(cvals, subject, vmin=0, vmax=1, cmap="RdBu_r"):
    import pymeg.source_reconstruction as pymegsr
    import pymeg.atlas_glasser as ag
    import cortex
    from scipy import misc

    labels = pymegsr.get_labels(
        subject=subject,
        filters=["*wang*.label", "*JWDG*.label"],
        annotations=["HCPMMP1"],
    )
    labels = pymegsr.labels_exclude(
        labels=labels,
        exclude_filters=[
            "wang2015atlas.IPS4",
            "wang2015atlas.IPS5",
            "wang2015atlas.SPL",
            "JWDG_lat_Unknown",
        ],
    )
    labels = pymegsr.labels_remove_overlap(
        labels=labels, priority_filters=["wang", "JWDG"]
    )

    V = ag.rois2vertex(
        "fsaverage", cvals, "lh", labels, vmin=vmin, vmax=vmax, cmap=cmap
    )
    cortex.quickflat.make_png(
        "/Users/nwilming/Desktop/test.png",
        V,
        cutout="left",
        with_colorbar=False,
        with_labels=False,
        with_rois=False,
    )
    return misc.imread("/Users/nwilming/Desktop/test.png")


"""
The next plots investigate SSD data -> encoding of contrast during stimulus phase.
"""


@memory.cache
def do_stats(x):
    from mne.stats import permutation_cluster_1samp_test

    return permutation_cluster_1samp_test(x, threshold=dict(start=0, step=0.2))


def ssd_overview_plot(ssd, area=["vfcPrimary", "JWG_M1"], ylim=[0, 0.1]):
    import matplotlib

    signals = ["SSD", "SSD_acc_contrast"]
    gs = matplotlib.gridspec.GridSpec(len(area), 2)
    for j, a in enumerate(area):
        for i, signal in enumerate(signals):
            ax = plt.subplot(gs[j, i])
            plot_ssd_per_sample(
                ssd.query('signal=="%s"' % signal), area=a, ax=ax, ylim=ylim
            )
            plt.title(signal)


def ssd_encoding_plot(ssd, ylim=[-0.01, 0.11]):
    import matplotlib

    signals = ["SSD", "SSD_acc_contrast"]
    gs = matplotlib.gridspec.GridSpec(2, 2)
    for i, signal in enumerate(signals):
        ax = plt.subplot(gs[0, i])
        plot_ssd_per_sample(
            ssd.Averaged.query('signal=="%s"' % signal),
            area="vfcPrimary",
            ax=ax,
            ylim=ylim,
        )
        plt.title(signal)
    for i, signal in enumerate(signals):
        ax = plt.subplot(gs[1, i])
        plot_ssd_per_sample(
            ssd.Lateralized.query('signal=="%s"' % signal),
            area="JWG_M1",
            ax=ax,
            ylim=ylim,
        )


def errorbar(x, y, xerr=None, yerr=None, color="b"):
    plt.plot(x, y, ".", color=color)
    if xerr is not None:
        for xx, o in zip(x, xerr):
            plt.plot([xx, xx], o, color=color, lw=0.5)
    if yerr is not None:
        for yy, o in zip(y, yerr):
            plt.plot(o, [yy, yy], color=color, lw=0.5)


def ssd_index_plot(
    idx, ssd, labels=None, rgb=True, brain=None, integration_slice=slice(0.4, 1.2)
):
    import seaborn as sns
    import matplotlib

    gs = matplotlib.gridspec.GridSpec(4, 2)
    if labels is None:
        labels = sr.get_labels(
            subject="fsaverage",
            filters=["*wang*.label", "*JWDG*.label"],
            annotations=["HCPMMP1"],
        )
        labels = sr.labels_exclude(
            labels=labels,
            exclude_filters=[
                "wang2015atlas.IPS4",
                "wang2015atlas.IPS5",
                "wang2015atlas.SPL",
                "JWDG_lat_Unknown",
            ],
        )
        labels = sr.labels_remove_overlap(
            labels=labels, priority_filters=["wang", "JWDG"]
        )
    plt.figure(figsize=(7.5, 11))
    avg = pd.pivot_table(
        idx.query('hemi=="Averaged"'), index="cluster", columns="subject"
    )
    lat = pd.pivot_table(
        idx.query('hemi=="Lateralized"'), index="cluster", columns="subject"
    )
    plt.subplot(gs[0, 0])  # 3,2,1)
    k = sps_2lineartime(get_ssd_per_sample(ssd.Averaged, "SSD", area="vfcPrimary"))
    ka = sps_2lineartime(
        get_ssd_per_sample(ssd.Averaged, "SSD_acc_contrast", area="vfcPrimary")
    )
    plt.plot(k.index, k.max(1), "b", label="Contrast")
    plt.plot(k.index, k, "b", lw=0.1)
    plt.plot(ka.index, ka.max(1), "r", label="Acc. contrast")
    plt.plot(ka.index, ka, "r", lw=0.1)
    plt.legend()
    yl = plt.ylim()
    plt.fill_between(
        [integration_slice.start, integration_slice.stop],
        [yl[0], yl[0]],
        [yl[1], yl[1]],
        color="k",
        alpha=0.25,
        zorder=-1,
        edgecolor=None,
    )
    plt.title("V1")
    plt.xlabel("time")
    plt.ylabel("encoding strength")
    plt.subplot(gs[0, 1])
    # plt.subplot(3,2, 2)
    k = sps_2lineartime(get_ssd_per_sample(ssd.Lateralized, "SSD", area="JWG_M1"))
    ka = sps_2lineartime(
        get_ssd_per_sample(ssd.Lateralized, "SSD_acc_contrast", area="JWG_M1")
    )
    plt.plot(k.index, k.max(1), "b", label="Contrast")
    plt.plot(k.index, k, "b", lw=0.1)
    plt.plot(ka.index, ka.max(1), "r", label="Acc. contrast")
    plt.plot(ka.index, ka, "r", label="Acc. contrast", lw=0.1)
    plt.xlabel("time")
    plt.ylabel("encoding strength")
    yl = plt.ylim()
    plt.title("M1")
    plt.fill_between(
        [integration_slice.start, integration_slice.stop],
        [yl[0], yl[0]],
        [yl[1], yl[1]],
        color="k",
        alpha=0.25,
        zorder=-1,
        edgecolor=None,
    )

    plt.subplot(gs[1, 0])
    plt.plot(avg.SSD.mean(1), avg.SSD_acc_contrast.mean(1), "m.", label="Averaged")
    plt.plot(lat.SSD.mean(1), lat.SSD_acc_contrast.mean(1), "c.", label="Lateralized")

    plt.legend(loc="upper right")
    plt.xlabel("Contrast enc.")
    plt.ylabel("Average contrast enc.")
    plt.plot(plt.xlim(), plt.xlim(), "k", lw=0.5)
    plt.xticks([0.01, 0.02, 0.03, 0.04])
    plt.yticks([0.01, 0.02, 0.03, 0.04])
    xl = plt.xlim()
    plt.ylim(xl)

    gs01 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1])

    palette = {}
    avg_index = avg.SSD - avg.SSD_acc_contrast
    lat_index = lat.SSD - lat.SSD_acc_contrast

    x = np.arange(avg_index.shape[0])
    d = avg.SSD - lat.SSD_acc_contrast  # (avg_index+lat_index)

    d_hpd = get_many_posterior_diff_one_group(d.T.values)
    d = d.mean(1)
    k = np.argsort(d)

    for name in d.index:
        # try:
        value = d[name]
        if rgb:
            import matplotlib

            norm = matplotlib.colors.Normalize(vmin=-0.04, vmax=0.04)
            cm = matplotlib.cm.get_cmap("RdBu_r")
            palette[name] = cm(norm(value))
        else:
            palette[name] = value

    # plt.plot(x, d[k], '.', color=sns.xkcd_rgb['yellow orange'])
    cnt = 0
    from brokenaxes import brokenaxes

    bax = plt.subplot(gs01[1, 0])
    cnt = 0
    for name, xx, val, o in zip(d[k].index, x, d[k], d_hpd[k, :]):
        if (xx > 9) and (xx < 58):
            continue
        bax.plot([cnt, cnt], o, color=palette[name], lw=1)
        bax.plot(cnt, val, ".", color=palette[name])
        cnt += 1

    d = 0.015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=bax.transAxes, color="k", clip_on=False)
    bax.plot(
        np.array([-d, +d]) + 0.5, 2 * np.array((-d, +d)) - 0.085, **kwargs
    )  # top-left diagonal
    # bax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    bax.axhline(0, color="k", lw=0.25)
    bax.set_xlabel("Area")
    bax.set_ylabel("Selectivity index")
    bax.set_xticks([])
    bax.set_yticks([-0.04, 0, 0.04])
    plt.subplot(gs01[0, 0])
    plt.plot(avg.SSD.mean(1), lat.SSD_acc_contrast.mean(1), ".")
    plt.ylabel("Accum. lat")
    plt.xlabel("Contr. avg")
    plt.ylim([-0.01, 0.05])
    plt.xlim([-0.01, 0.08])

    sns.despine(offset=5)

    plt.subplot(gs[2:, :])
    # plt.subplot(3,2,5)
    if brain is None:
        from conf_analysis.meg import srtfr

        del palette["HCPMMP1_insular_front_opercular"]
        del palette["HCPMMP1_cingulate_pos"]
        del palette["HCPMMP1_frontal_orbital_polar"]
        del palette["HCPMMP1_frontal_inferior"]
        del palette["HCPMMP1_dlpfc"]
        # del palette['HCPMMP1_premotor']
        del palette["HCPMMP1_paracentral_midcingulate"]
        del palette["post_medial_frontal"]
        del palette["vent_medial_frontal"]
        del palette["ant_medial_frontal"]
        brain = plot_brain_color_legend(palette, srtfr.get_clusters())
    img = brain.save_montage(
        "/Users/nwilming/Desktop/acc_vs_ssd_spec.png", ["lat", "pa"]
    )
    plt.imshow(img, aspect="equal")
    plt.xticks([])
    plt.yticks([])
    sns.despine(left=True, bottom=True, ax=plt.gca())
    plt.tight_layout()
    return palette, brain


def view_surf(palette, labels):
    import nilearn
    from functools import reduce
    from conf_analysis.meg import srtfr
    from pymeg import atlas_glasser as ag
    import mne

    cluster, _, _, _ = ag.get_clusters()
    cluster.update(srtfr.get_clusters())

    coords, faces = nilearn.surface.load_surf_mesh(
        "/Users/nwilming/u/freesurfer_subjects/fsaverage/surf/lh.pial"
    )
    colors = np.ones(len(coords)) * 0
    for name, color in palette.items():
        try:
            ls = cluster[name]

            ls = [l for l in labels if any([x in l.name for x in ls])]
            label = reduce(lambda x, y: x + y, ls)
            try:
                vertices = label.lh.vertices
            except AttributeError:
                if "lh" in label.name:
                    vertices = label.vertices
            # vertices = label.lh.vertices
            colors[vertices] = color
        except KeyError:
            print("Nothing for ", name)
    return [coords, faces], colors


def get_label_in_mni(name, labels):
    from functools import reduce
    from conf_analysis.meg import srtfr
    from pymeg import atlas_glasser as ag
    import mne

    cluster, _, _, _ = ag.get_clusters()
    cluster.update(srtfr.get_clusters())

    ls = cluster[name]
    ls = [l for l in labels if any([x in l.name for x in ls])]
    label = reduce(lambda x, y: x + y, ls)
    try:
        return mne.vertex_to_mni(label.lh.vertices, 0, label.subject).mean(0)
    except AttributeError:
        if "lh" in label.name:
            return mne.vertex_to_mni(label.vertices, 0, label.subject).mean(0)
        else:
            return mne.vertex_to_mni(label.vertices, 1, label.subject).mean(0)


@memory.cache()
def get_ssd_idx(ssd, integration_slice=slice(0.4, 1.2), pair=False):
    if not pair:
        ssd_indices = (
            ssd.Averaged.stack()
            .groupby(["subject", "signal", "cluster"])
            .apply(get_ssd_index)
            .unstack("signal")
        )
    else:
        ssd_indices = (
            ssd.Pair.stack()
            .groupby(["subject", "signal", "cluster"])
            .apply(get_ssd_index)
            .unstack("signal")
        )

    ssd_indices.loc[:, "SSDvsACC"] = (
        ssd_indices.loc[:, "SSD"] - ssd_indices.loc[:, "SSD_acc_contrast"]
    )
    if not pair:
        ssd_indices.loc[:, "hemi"] = "Averaged"
    else:
        ssd_indices.loc[:, "hemi"] = "Pair"
    ssd_indices.set_index("hemi", append=True, inplace=True)

    if not pair:
        ssd_indices2 = (
            ssd.Lateralized.stack()
            .groupby(["subject", "signal", "cluster"])
            .apply(get_ssd_index)
            .unstack("signal")
        )
    else:
        ssd_indices2 = (
            ssd.Pair.stack()
            .groupby(["subject", "signal", "cluster"])
            .apply(get_ssd_index)
            .unstack("signal")
        )
    ssd_indices2.loc[:, "SSDvsACC"] = (
        ssd_indices2.loc[:, "SSD"] - ssd_indices2.loc[:, "SSD_acc_contrast"]
    )
    if not pair:
        ssd_indices2.loc[:, "hemi"] = "Lateralized"
    else:
        ssd_indices2.loc[:, "hemi"] = "Pair"
    ssd_indices2.set_index("hemi", append=True, inplace=True)
    return pd.concat([ssd_indices, ssd_indices2])


def get_ssd_index(ssd, signal=None, area=None, integration_slice=slice(0.4, 1.2)):
    if signal is None:
        signal = ssd.index.get_level_values("signal").unique()[0]
    sps = sps_2lineartime(get_ssd_per_sample(ssd, signal, area))
    return get_max_encoding(sps, lim=integration_slice)


def get_ssd_per_sample(ssd, signal, area=None):
    """
    Make a matrix that encodes SSD per sample and latency.
    """
    try:
        k = ssd.query('signal=="%s"' % signal).groupby(["sample", "latency"]).mean()
    except AttributeError:
        ssd = ssd.to_frame()
        k = ssd.query('signal=="%s"' % signal).groupby(["sample", "latency"]).mean()
    if area is None:
        area = k.columns.values[0]
        if len(area) > 1:
            # Dealing with a multi index, which is problematic for
            # pivot_table. Assume cluster is last
            area = area[-1]
            k.columns = [area]

    k = pd.pivot_table(k, columns="latency", index="sample", values=area)
    return k


def sps_2lineartime(sps):
    """
    Take a SPS and shift each sample in time to correct 
    position in trial.
    """
    rows = []
    for sample in np.arange(10):
        r = sps.loc[sample, :].T.to_frame().reset_index()
        r.loc[:, "latency"] = np.around(r.loc[:, "latency"] + sample * 0.1, 3)
        rows.append(r.set_index("latency"))
    return pd.concat(rows, sort=True, axis=1)


def get_max_encoding(sps, lim=slice(0.4, 1.2)):
    sps = sps.loc[lim, :].T.max()
    return np.mean(sps.values)  # np.trapz(sps.values, sps.index.values)


def plot_ssd_per_sample(
    ssd,
    area="vfcvisual",
    cmap="magma",
    alpha=0.05,
    latency=0.18,
    save=False,
    ax=None,
    stats=True,
    ylim=[0, 0.1],
):
    """
    """
    import pylab as plt
    import seaborn as sns

    sns.set_style("ticks")
    cmap = plt.get_cmap(cmap)
    if ax is None:
        plt.figure(figsize=(6, 3.3))
        ax = plt.gca()

    for sample, ds in ssd.groupby("sample"):
        ds = ds.astype(float)
        if ds.columns.is_categorical():
            # Convert index to non-categorical to avoid pandas
            # bug? #19136
            ds.columns = pd.Index([x for x in ds.columns.values])
        k = ds.groupby(["subject", "latency"]).mean().reset_index()
        k = pd.pivot_table(
            ds.reset_index(), columns="latency", index="subject", values=area
        ).dropna()
        baseline = k.loc[:, :0].mean(axis=1)
        baseline_corrected = k.sub(baseline, axis=0)

        ax.plot(
            k.columns.values + 0.1 * sample, k.values.mean(0), color=cmap(sample / 10.0)
        )
        ax.set_ylim(ylim)
        if stats:
            t_obs, clusters, cluster_pv, H0 = do_stats(baseline_corrected.values)
            sig_x = (k.columns.values + 0.1 * sample)[cluster_pv < alpha]
            ax.plot(
                sig_x, sig_x * 0 - 0.0001 * np.mod(sample, 2), color=cmap(sample / 10)
            )
        ax.axvline(0.1 * sample + latency, color="k", alpha=0.25, zorder=-1, lw=1)
    ax.set_xlabel(r"$time$")
    ax.set_ylabel(r"$contrast \sim power$")
    sns.despine(trim=True, ax=ax)
    if save:
        plt.tight_layout()
        plt.savefig("/Users/nwilming/Dropbox/UKE/confidence_study/ssd_slopes_corr.svg")



def get_ssd_kernel_data(peaks):
    pass


def plot_ssd_peaks(peaks, palette, K=None):
    import pylab as plt
    import seaborn as sns
    from adjustText import adjust_text

    # from scipy.stats import
    plt.figure(figsize=(5.5, 4.5))
    pvt = peaks  # pd.pivot_table(peaks.astype(
    # float), index='sample', columns='subject')
    texts = []
    lines = []
    _, visual_field_clusters, _, _ = ag.get_clusters()
    for key, color in palette.items():
        try:
            apvt = pvt.loc[:, key]
        except KeyError:
            continue
        x, y = apvt.index.values, apvt.mean(1).values
        # Simple t-tests
        p = 1.0
        if K is not None:
            from scipy.stats import ttest_1samp

            Y = apvt.values
            corrs = []
            for sub in range(1, 16):
                corrs.append(
                    np.corrcoef(K.loc[:, sub].values, apvt.loc[:, sub].values)[0, 1]
                )
            _, p = ttest_1samp(np.tanh(corrs), 0)

        ps = plt.plot(x, y, color=color)

        if (key in visual_field_clusters) or ("cingulate_pos" in key):
            key = key.replace("vfc", "")
            key = key.replace("HCPMMP1_", "")
            if p < 0.05:
                key += "*"
            t = plt.text(x[-1], y[-1], key, color=color, size=10)

        texts.append(t)
        lines.append(ps[0])

    plt.xlim(-1, 11)
    adjust_text(
        texts,
        only_move={"text": "x", "objects": "x"},
        add_objects=lines,
        ha="center",
        va="bottom",
    )
    plt.xlabel(r"$sample$")
    plt.ylabel(r"$slope$")
    sns.despine(trim=True)
    ax2 = plt.gca().twinx()
    ax2.plot(K.index.values, K.mean(1), "k")
    ax2.set_ylim([-0.1, 0.22])
    ax2.set_yticks([])
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig("/Users/nwilming/Dropbox/UKE/confidence_study/slopes_at_peakT_SSD.svg")


def extract_latency_peak_slope(ssd):
    ssd = ssd.query('epoch=="stimulus" & latency>=0 & Classifier=="Ridge"')
    ps = []
    index = []
    for idx, d in ssd.astype(float).groupby(["subject", "sample"]):
        levels = list(set(ssd.index.names) - set(["latency"]))

        d.index = d.index.droplevel(levels)
        d = d.idxmax()
        ps.append(d)
        index.append(idx)
    ps = pd.concat(ps, 1).T
    return ps.set_index(pd.MultiIndex.from_tuples(index, names=["subject", "sample"]))


def extract_peak_slope(ssd, latency=0.18, dt=0.01, peak_latencies=None):
    if peak_latencies is None:
        print('Single latency:', latency)
        ssd = ssd.query(
            'epoch=="stimulus" & %f<=latency & %f<=latency & Classifier=="Ridge"'
            % (latency - dt, latency + dt)
        )

        ssd = ssd.astype(float).groupby(["subject", "sample"]).max()
        return pd.pivot_table(ssd, index="sample", columns="subject")
    else:
        reslist = []
        assert len(ssd.index.get_level_values("signal").unique()) == 1
        for idx, ds in ssd.groupby(["subject", "sample"]):
            peak_idx = peak_latencies.loc[idx, :]
            levels = list(set(ds.index.names) - set(["latency"]))
            ds.index = ds.index.droplevel(levels)
            res = {"subject": idx[0], "sample": idx[1]}
            for col in ds.columns:
                latency = peak_idx.loc[col]
                value = ds.loc[latency, col]
                res.update({col: value})
            reslist.append(res)
        peaks = pd.DataFrame(reslist)
        peaks.set_index(["subject", "sample"], inplace=True)
        return peaks  # pd.pivot_table(ssd, index='sample', columns='subject')


def extract_kernels(dz, contrast_mean=0.0, include_ref=False):
    """Kernels for each subject"""
    from conf_analysis.behavior import empirical

    K = (
        dz.groupby(["snum"])
        .apply(
            lambda x: empirical.get_pk(
                x, contrast_mean=contrast_mean, response_field="response",
                include_ref=include_ref
            )
        )
        .groupby(level=["snum", "time"])
        .apply(lambda x: (x.query("optidx==1").mean() - x.query("optidx==0").mean()))
    )
    K.index.names = ["subject", "sample"]
    ks = pd.pivot_table(K, values="contrast", index="sample", columns="subject")
    return ks


def plot_cluster_overview(decoding, ssd, tfr_resp, peaks, kernel, nf=True):
    """
    Input should be a series.

    2 x 3 plot with
                SSD     TFR     Decoding
    stimulus
    response
    """
    if nf:
        plt.figure(figsize=(12, 5))
    from conf_analysis.meg import srtfr
    from scipy.stats import linregress

    area = ssd.name
    assert decoding.name == area
    gs = gridspec.GridSpec(2, 3)
    tslice = {"stimulus": slice(-0.35, 1.35), "response": slice(-1, 0.5)}
    xticks = {"stimulus": [-0.35, 0, 0.5, 1, 1.35], "response": [-1, -0.5, 0, 0.5]}
    cbars = []
    for i, epoch in enumerate(["stimulus", "response"]):
        if epoch == "stimulus":
            plt.subplot(gs[i, 0])
            # SSD x stimulus
            plot_ssd_per_sample(ssd, area=area, cmap="magma", ax=plt.gca())
            ps = peaks.loc[:, area]
            K = kernel.mean(1).values
            P = ps.mean(1).values
            plt.plot(0.183 + (ps.index.values / 10.0), P, "k", alpha=0.5)
            slope, inter, _, _, _ = linregress(K, P)
            plt.plot(0.183 + (kernel.index.values / 10.0), slope * K + inter)

        plt.subplot(gs[i, 1])
        id_epoch = tfr_resp.index.get_level_values("epoch") == epoch

        s = srtfr.get_tfr_stack(tfr_resp.loc[id_epoch], area, tslice=tslice[epoch])
        t, p, H0 = srtfr.stats_test(s)
        p = p.reshape(t.shape)
        cbar = srtfr.plot_tfr(
            tfr_resp.loc[id_epoch],
            area,
            ps=p,
            minmax=5,
            title_color=None,
            tslice=tslice[epoch],
        )
        if epoch == "stimulus":
            plt.axvline(-0.25, ls="--", color="k", alpha=0.5)
            plt.axvline(0.0, ls="--", color="k", alpha=0.5)
        if epoch == "response":
            plt.xlabel(r"$time$")
        cbars.append(cbar)
        plt.xticks(xticks[epoch])
        plt.subplot(gs[i, 2])
        signals = pd.pivot_table(
            decoding.reset_index().query('epoch=="%s"' % epoch),
            columns="signal",
            index="latency",
            values=area,
        ).loc[tslice[epoch]]

        for col in signals:
            plt.plot(signals[col].index.values, signals[col].values, label=col)
        plt.ylabel(r"$AUC$")
        plt.xticks(xticks[epoch])
        plt.ylim([0.25, 0.75])
        if epoch == "response":
            plt.xlabel(r"$time$")
    plt.legend()
    sns.despine(trim=True)
    for cbar in cbars:
        cbar.ax.yaxis.set_ticks_position("right")
    return cbar


def compare_kernel(K, peaks):
    res = {}
    for area in peaks.columns.get_level_values("cluster").unique():
        y = peaks[area]
        res[area] = [np.corrcoef(y.loc[:, i], K.loc[:, i])[0, 1] for i in K.columns]
    return res


def get_kernel_fits(kernel, peaks):
    from scipy.optimize import curve_fit
    from scipy.stats import linregress

    def func(x, a, c, d):
        return a * np.exp(-c * x) + d

    pars = []
    for sub in peaks.columns:
        x = np.arange(10)
        yk = kernel.loc[:, sub]  # - kernel.loc[:, sub].mean()
        yp = peaks.loc[:, sub]  # - peaks.loc[:, sub].mean()
        slope, inter, _, _, _ = linregress(yk, yp)
        yk = slope * yk + inter
        popt_k, pcov = curve_fit(func, x, yk, p0=(1, 1e-6, 1), maxfev=4000)
        popt_p, pcov = curve_fit(func, x, yp, p0=(1, 1e-6, 1), maxfev=4000)
        plt.subplot(3, 5, sub)
        plt.plot(x, yk, "r")
        plt.plot(x, func(x, *popt_k), "r--", alpha=0.5)
        plt.subplot(3, 5, sub)
        plt.plot(x, yp, "b")
        plt.plot(x, func(x, *popt_p), "b--", alpha=0.5)
        results = {
            "subject": sub,
            "Ka": popt_k[0],
            "Kc": popt_k[1],
            "Kd": popt_k[2],
            "Pa": popt_p[0],
            "Pc": popt_p[1],
            "Pd": popt_p[2],
        }
        pars.append(results)
    return pd.DataFrame(pars)


def make_r_data(area, peaks, k):
    ps = peaks.stack()
    ks = k.stack()
    ks.name = "kernel"
    psks = ps.join(ks)
    P = psks.loc[:, ("kernel", area)].reset_index()
    return P


def fit_correlation_model(data, area):
    """
    Use R to fit a bayesian hierarchical model that estimates correlation
    between time constants of kernels.
    """
    from rpy2.robjects import r, pandas2ri

    pandas2ri.activate()
    from rpy2.robjects.packages import importr

    importr("brms")
    code = """
        prior1 <- (prior(normal(.2, 1), nlpar = "b1", resp = "kernel")
                  + prior(normal(0, 2), nlpar = "b2", resp="kernel")
                  + prior(normal(.2, 1), nlpar = "b1", resp = "{area}")
                  + prior(normal(0, 2), nlpar = "b2", resp="{area}"))

        FF <- bf(cbind(kernel, {area}) ~ b1 * exp(b2 * sample),
                b1 ~ 1|subject,
                b2~1|2|subject,
                nl = TRUE)
        fit1 <- brm(FF, data = P, prior = prior1,
                    control=list(adapt_delta=0.99),
                    cores=1, chains=2, iter=1000)
        random <- ranef(fit1)
        fixed <- fixef(fit1)
        summary(fit1)
    """.format(
        area=area
    )
    print(code)
    r.assign("P", data)
    df = r(code)
    return df, r


def plot_brain_color_legend(palette, brain=None, ogl=False, subject="fsaverage"):
    """
    Plot all ROIs on pysurfer brain. Colors given by palette.
    """
    from surfer import Brain
    from pymeg import atlas_glasser as ag

    if not ogl:
        labels = sr.get_labels(
            subject=subject,
            filters=["*wang*.label", "*JWDG*.label"],
            annotations=["HCPMMP1"],
        )
        labels = sr.labels_exclude(
            labels=labels,
            exclude_filters=[
                "wang2015atlas.IPS4",
                "wang2015atlas.IPS5",
                "wang2015atlas.SPL",
                "JWDG_lat_Unknown",
            ],
        )
        from conf_analysis.meg import srtfr

        clusters = srtfr.get_clusters()
        labels = sr.labels_remove_overlap(
            labels=labels, priority_filters=["wang", "JWDG"]
        )
        lc = ag.labels2clusters(labels, clusters)
    else:
        from conf_analysis.meg import srtfr

        labels = sr.get_labels(subject=subject, filters=[], annotations=["HCPMMP1"])
        clusters = srtfr.get_ogl_clusters()
        lc = ag.labels2clusters(labels, clusters)
    if brain is None:
        brain = Brain(subject, "lh", "inflated", views=["lat"], background="w")
    for cluster, labelobjects in lc.items():
        if cluster in palette.keys():
            color = palette[cluster]
            for l0 in labelobjects:
                if l0.hemi == "lh":
                    brain.add_label(l0, color=color, alpha=1)

    return brain


def plot_brain_color_annotations(
    palette, brain=None, low=0.3, high=0.7, alpha=0.9, colormap="RdBu_r", thresh=0
):
    """
    Plot all ROIs on pysurfer brain. Colors given by palette.

    Assume that labels match annotation HCPMMP1
    """
    from surfer import Brain
    import nibabel as nib
    from pymeg import atlas_glasser as ag

    labels, cmap, names = nib.freesurfer.read_annot(
        "/Users/nwilming/u/freesurfer_subjects/fsaverage/label/lh.HCPMMP1.annot",
        orig_ids=True,
    )
    data = labels.astype(float) * 0 + -1
    non_itms = {}
    for roi, color in palette.items():
        idx = np.where(
            [("L_" + roi + "_ROI") in l.decode("utf-8") for l in np.array(names)]
        )[0]
        if len(idx) == 1:
            tval = cmap[idx[0], -1]
            # print(any(labels==tval))
            data[labels == tval] = color
            # print(np.unique(data))
            pass
        elif len(idx) == 0:
            print("No annot for key %s" % roi)
            non_itms[roi] = color
        else:
            raise RuntimeError("Non unique ROI name in palette")

    if brain is None:
        brain = Brain("fsaverage", "lh", "inflated", views=["lat"], background="w")
    if not all(data==-1):
        brain.add_data(
            data, low, high, thresh=thresh, colormap=colormap, alpha=alpha, colorbar=False
        )

    return brain, non_itms


@memory.cache
def get_posterior_diff(data, ref_data):
    """
    Get uncertainty around average mean by means of bayesian 
    inference. For AUC values, nothing else.
    """
    import pymc3 as pm

    with pm.Model():
        mu_d = pm.Normal("mud", 0.5, 1)
        std_d = pm.Uniform("stdd", lower=0, upper=1)
        mu_r = pm.Normal("mur", 0.5, 1)
        std_r = pm.Uniform("stdr", lower=0, upper=1)
        v = pm.Exponential("ν_minus_one", 1 / 29.0) + 1
        pm.StudentT("OutD", mu=mu_d, lam=std_d ** -2, nu=v, observed=data)
        pm.StudentT("OutR", mu=mu_r, lam=std_r ** -2, nu=v, observed=ref_data)
        diffs = pm.Deterministic("mu_diff", mu_d - mu_r)
        k = pm.sample()
    mud = k.get_values("mud")
    mur = k.get_values("mur")
    mudiff = k.get_values("mu_diff")
    return mud.mean(0), mur.mean(0), pm.stats.hpd(mudiff)


@memory.cache
def get_many_posterior_diff_one_group(data, ref_data):
    """
    Get uncertainty around average mean by means of bayesian 
    inference. For AUC values, nothing else.
    """
    import pymc3 as pm

    n = data.shape[1]

    with pm.Model():
        mu_d = pm.Normal("mud", data.mean(0), data.std(0) * 5, shape=n)
        std_d = pm.Uniform("stdd", lower=0, upper=data.std(0) * 20, shape=n)
        mu_r = pm.Normal("mur", ref_data.mean(0), data.std(0) * 5, shape=n)
        std_r = pm.Uniform("stdr", lower=0, upper=data.std(0) * 20, shape=n)
        # v = pm.Exponential("ν_minus_one", 1 / 29.0, shape=n) + 1
        # pm.StudentT("OutD", mu=mu_d, lam=std_d ** -2, nu=v, observed=data, shape=n)
        # pm.StudentT("OutR", mu=mu_r, lam=std_r ** -2, nu=v, observed=ref_data, shape=n)
        pm.Normal("OutD", mu_d, std_d, observed=data, shape=n)
        pm.Normal("OutR", mu_r, std_r, observed=ref_data, shape=n)
        diffs = pm.Deterministic("mu_diff", mu_d - mu_r)
        k = pm.sample(tune=1500, draws=1500)
    mud = k.get_values("mud")
    mur = k.get_values("mur")
    mudiff = k.get_values("mu_diff")
    return pm.stats.hpd(mud), pm.stats.hpd(mur), pm.stats.hpd(mudiff)


@memory.cache
def get_many_posterior_diff_one_group(data):
    """
    Get uncertainty around average mean by means of bayesian 
    inference. For AUC values, nothing else.
    """
    import pymc3 as pm

    n = data.shape[1]

    with pm.Model():
        mu_d = pm.Normal("mud", data.mean(0), data.std(0) * 10, shape=n)
        std_d = pm.Uniform("stdd", lower=0, upper=data.std(0) * 20, shape=n)
        # v = pm.Exponential("ν_minus_one", 1 / 29.0, shape=n) + 1
        # pm.StudentT("OutD", mu=mu_d, lam=std_d ** -2, nu=v, observed=data, shape=n)
        # pm.StudentT("OutR", mu=mu_r, lam=std_r ** -2, nu=v, observed=ref_data, shape=n)
        pm.Normal("OutD", mu_d, std_d, observed=data, shape=n)
        k = pm.sample(tune=1500, draws=1500)
    mud = k.get_values("mud")
    return pm.stats.hpd(mud)


def combine_and_save_var_SSD_signals():
    from itertools import product
    signals = ['SSD_acc_contrast', 'SSD']
    epochs = ['response', 'stimulus']
    df = pd.concat([
        combine_signals(signal, epoch) for signal, epoch in product(signals, epochs)
        ])
    df.to_hdf('/home/nwilming/conf_analysis/results/ssd_decoding_by_var_extra_analysis_20191203.hdf', 'df')
    return df

def combine_signals(signal, epoch):
    import glob
    globstr = '/mnt/homes/home028/nwilming/conf_meg/sr_decoding/concat_S*-%s_*var*%s-decoding.hdf'%(signal, epoch)
    print(globstr)
    files = glob.glob(globstr)
    frames = []
    for f in files:
        df = pd.read_hdf(f)
        cluster = [x.replace('[','').replace(']', '').replace("'","") for x in df.cluster.values]
        df.loc[:, 'cluster'] = cluster
        df.loc[:, 'signal'] = signal
        df.loc[:, 'epoch'] = epoch
        if 'varlow' in f:
            df.loc[:, 'variance'] = 'low'            
        elif 'varhigh' in f:
            df.loc[:, 'variance'] = 'high'
        elif 'varmed' in f:
            df.loc[:, 'variance'] = 'medium'

        #df.loc[:, 'subject'] = df.loc[:, 'sub']
        frames.append(df)
    return pd.concat(frames)
