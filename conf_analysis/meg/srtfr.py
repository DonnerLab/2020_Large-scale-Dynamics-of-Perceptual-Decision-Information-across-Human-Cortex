import os
import pandas as pd
from conf_analysis.meg import preprocessing
from pymeg.contrast_tfr import Cache, compute_contrast, augment_data
from pymeg import contrast_tfr
from pymeg import parallel
from joblib import Memory
import logging

logging.getLogger().setLevel(logging.INFO)
logger = contrast_tfr.logging.getLogger()
logger.setLevel(logging.INFO)


if "TMPDIR" in os.environ.keys():
    data_path = "/nfs/nwilming/MEG/"
else:
    data_path = "/home/nwilming/conf_meg/"

memory = Memory(location=os.environ["PYMEG_CACHE_DIR"], verbose=0)


contrasts = {
    "all": (["all"], [1]),
    "choice": (["hit", "fa", "miss", "cr"], (1, 1, -1, -1)),
    "stimulus": (["hit", "fa", "miss", "cr"], (1, -1, 1, -1)),
    "hand": (["left", "right"], (1, -1)),
    "confidence": (
        [
            "high_conf_high_contrast",
            "high_conf_low_contrast",
            "low_conf_high_contrast",
            "low_conf_low_contrast",
        ],
        (1, 1, -1, -1),
    ),
    # (HCHCont - LCHCont) + (HCLCont - LCLCont) ->
    #  HCHCont + HCLcont  -  LCHCont - LCLCont
    # Example:
    # HCm = 10, LCm = 7 (10-7) + (1-4) = 3-3 == 0
    "confidence_asym": (
        [
            "high_conf_high_contrast",
            "high_conf_low_contrast",
            "low_conf_high_contrast",
            "low_conf_low_contrast",
        ],
        (1, -1, -1, 1),
    ),
    # (HCHCont - LCHCont) - (HCLCont - LCLCont) ->
    #  HCHCont - HCLcont  -  LCHCont + LCLCont
}


def submit_contrasts(collect=False):
    import numpy as np
    import time

    tasks = []
    for subject in range(1, 16):
        tasks.append((contrasts, subject))
    res = []
    for cnt, task in enumerate(tasks):
        try:
            r = _eval(
                get_contrasts,
                task,
                collect=collect,
                walltime="01:30:00",
                tasks=9,
                memory=70,
            )
            res.append(r)
            #if np.mod(cnt + 1, 10)==0:
            #    if not collect:
            #        time.sleep(60 * 10)
        except RuntimeError:
            print("Task", task, " not available yet")
    return res


def _eval(func, args, collect=False, **kw):
    """
    Intermediate helper to toggle cluster vs non cluster
    """
    if not collect:
        if not func.in_store(*args):
            print("Submitting %s to %s for parallel execution" % (len(args), func))
            print(args)
            parallel.pmap(func, [args], **kw)
    else:
        if func.in_store(*args):
            print("Submitting %s to %s for collection" % (str(args), func))
            df = func(*args)
            return df
        else:
            raise RuntimeError("Result not available.")


def get_clusters():
    """
    Pimp cluster defs
    """
    from pymeg import atlas_glasser as ag
    # fmt: off
    areas = ["47s", "47m", "a47r", "11l", "13l", "a10p", "p10p", "10pp", "10d", "OFC",
             "pOFC", "44", "45", "IFJp", "IFJa", "IFSp", "IFSa", "47l", "p47r", "8C", "8Av",
             "i6-8", "s6-8", "SFL", "8BL", "9p", "9a", "8Ad", "p9-46v", "a9-46v", "46",
             "9-46d", "SCEF", "p32pr", "a24pr", "a32pr", "p24", "p32", "s32", "a24", "10v",
             "10r", "25", "d32", "8BM", "9m"]
    # fmt: on

    areas = {
        "NSWFRONT_" + area: ["L_{}_ROI-lh".format(area), "L_{}_ROI-lh".format(area)]
        for area in areas
    }
    all_clusters, _, _, _ = ag.get_clusters()
    all_clusters.update(areas)

    all_clusters = {
        k: v
        for k, v in all_clusters.items()
        if (k.startswith("NSWFRONT")) or (k in ag.areas.values())
    }
    return all_clusters


def get_ogl_clusters():
    """
    Pimp cluster defs
    """
    from pymeg import atlas_glasser as ag
    # fmt: off
    areas = ["TE1a", "VVC", "FOP1", "10v", "6r", "H", "LIPv", "OFC", "PFop", "STSvp",
             "VMV1", "STGa", "p24pr", "TGv", "3a", "p9-46v", "9m", "IFJp", "LIPd", "pOFC",
             "IPS1", "7PC", "PIT", "V1", "SCEF", "i6-8", "25", "PoI2", "a24pr", "8Av",
             "V2", "p47r", "V4", "p10p", "10d", "3b", "a24", "TA2", "10pp", "AIP",
             "PCV", "6d", "TF", "31pd", "FOP5", "MST", "IP1", "LO3", "PH", "45",
             "8Ad", "s6-8", "VMV2", "a47r", "46", "a9-46v", "FOP2", "V3CD", "PEF", "ProS",
             "p24", "MI", "PreS", "STSdp", "a10p", "MBelt", "FFC", "VMV3", "V3A", "VIP",
             "PoI1", "TE2p", "52", "9a", "31pv", "PeEc", "PI", "IFSp", "4", "A4",
             "AAIC", "RI", "5m", "23c", "7m", "PGp", "PHT", "p32", "6a", "FOP4",
             "PGi", "47l", "PGs", "MT", "55b", "A1", "TPOJ2", "Pir", "PHA3", "LO2",
             "IFSa", "p32pr", "8C", "LO1", "d32", "V3B", "V3", "V7", "6mp", "AVI",
             "d23ab", "31a", "DVT", "47m", "PFt", "9-46d", "A5", "V4t", "TPOJ1", "1",
             "PF", "6v", "PBelt", "OP2-3", "PHA2", "V8", "V6", "STSva", "44", "v23ab",
             "7Pm", "24dd", "TE1p", "OP4", "TE1m", "2", "V6A", "8BM", "IFJa", "10r",
             "IP0", "43", "OP1", "TE2a", "7Am", "6ma", "PFcm", "47s", "TPOJ3", "33pr",
             "FEF", "STSda", "MIP", "23d", "13l", "PHA1", "Ig", "24dv", "11l", "a32pr",
             "FST", "s32", "STV", "5mv", "9p", "TGd", "RSC", "POS1", "PFm", "IP2",
             "EC", "POS2", "FOP3", "LBelt", "PSL", "SFL", "5L", "7AL", "7PL", "9m"]
    areas = ["7AL", "7PL", "9m", "8BL"]
    # fmt: on

    areas = {
        area: ["L_{}_ROI-lh".format(area), "R_{}_ROI-rh".format(area)]
        for area in areas
    }

    return areas


@memory.cache(ignore=["scratch"])
def get_contrasts(contrasts, subject, baseline_per_condition=False, scratch=False):
    if subject <= 8:
        hemi = "rh_is_ipsi"
    else:
        hemi = "lh_is_ipsi"
    hemis = [hemi, "avg"]
    

    new_contrasts = {}
    for key, value in contrasts.items():
        new_contrasts[key+'lat'] = [value[0], value[1], hemi]
        new_contrasts[key+'avg'] = [value[0], value[1], 'avg']
    contrasts = new_contrasts
    print(contrasts)

    from os.path import join

    stim, resp = [], []
    for session in range(0, 4):
        stim.append(
            join(data_path, "sr_labeled/S%i-SESS%i-stimulus*.hdf" % (subject, session))
        )
        resp.append(
            join(data_path, "sr_labeled/S%i-SESS%i-response*.hdf" % (subject, session))
        )

    if scratch:
        from subprocess import run
        import os

        tmpdir = os.environ["TMPDIR"]
        command = "cp {stim} {tmpdir} & cp {resp} {tmpdir}".format(
            stim=stim, resp=resp, tmpdir=tmpdir
        )
        logging.info("Copying data with following command: %s" % command)
        p = run(command, shell=True, check=True)
        stim = join(data_path, tmpdir, "S%i-SESS%i-stimulus*.hdf" % (subject, session))
        resp = join(data_path, tmpdir, "S%i-SESS%i-response*.hdf" % (subject, session))
        logging.info("Copied data")

    meta = preprocessing.get_meta_for_subject(subject, "stimulus")
    response_left = meta.response == 1
    stimulus = meta.side == 1
    meta = augment_data(meta, response_left, stimulus)
    meta["high_conf_high_contrast"] = (meta.confidence == 2) & (meta.mc > 0.5)
    meta["high_conf_low_contrast"] = (meta.confidence == 2) & (meta.mc <= 0.5)
    meta["low_conf_high_contrast"] = (meta.confidence == 1) & (meta.mc > 0.5)
    meta["low_conf_low_contrast"] = (meta.confidence == 1) & (meta.mc <= 0.5)
    cps = []
    with Cache() as cache:
        try:
            contrast = compute_contrast(
                contrasts,
                #hemis,
                stim,
                stim,
                meta,
                (-0.25, 0),
                baseline_per_condition=baseline_per_condition,
                n_jobs=1,
                cache=cache,
                all_clusters=get_clusters(),
            )
            contrast.loc[:, "epoch"] = "stimulus"
            cps.append(contrast)
        except ValueError as e:
            # No objects to concatenate
            print(e)
            pass
        try:
            contrast = compute_contrast(
                contrasts,
                #hemis,
                resp,
                stim,
                meta,
                (-0.25, 0),
                baseline_per_condition=baseline_per_condition,
                n_jobs=1,
                cache=cache,
            )
            contrast.loc[:, "epoch"] = "response"
            cps.append(contrast)
        except ValueError as e:
            # No objects to concatenate
            print(e)
            pass
    contrast = pd.concat(cps)
    del cps
    contrast.loc[:, "subject"] = subject

    contrast.set_index(
        ["subject", "contrast", "hemi", "epoch", "cluster"], append=True, inplace=True
    )
    return contrast


def plot_mosaics(df, stats=False):
    import pylab as plt
    from pymeg.contrast_tfr import plot_mosaic

    for epoch in ["stimulus", "response"]:
        for contrast in [
            "all",
            "choice",
            "confidence",
            "confidence_asym",
            "hand",
            "stimulus",
        ]:
            for hemi in [True, False]:
                plt.figure()
                query = 'epoch=="%s" & contrast=="%s" & %s(hemi=="avg")' % (
                    epoch,
                    contrast,
                    {True: "~", False: ""}[hemi],
                )
                d = df.query(query)
                plot_mosaic(d, epoch=epoch, stats=stats)
                plt.suptitle(query)
                plt.savefig(
                    "/Users/nwilming/Desktop/tfr_average_%s_%s_lat%s.pdf"
                    % (epoch, contrast, hemi)
                )
                plt.savefig(
                    "/Users/nwilming/Desktop/tfr_average_%s_%s_lat%s.svg"
                    % (epoch, contrast, hemi)
                )


# Ignore following for now


def submit_stats(
    contrasts=["all", "choice", "confidence", "confidence_asym", "hand", "stimulus"],
    collect=False,
):
    all_stats = {}
    tasks = []
    for contrast in contrasts:
        for hemi in [True, False]:
            for epoch in ["stimulus", "response"]:
                tasks.append((contrast, epoch, hemi))
    res = []
    for task in tasks[:]:
        try:
            r = _eval(
                precompute_stats,
                task,
                collect=collect,
                walltime="08:30:00",
                tasks=2,
                memory=20,
            )
            res.append(r)
        except RuntimeError:
            print("Task", task, " not available yet")
    return res


@memory.cache()
def precompute_stats(contrast, epoch, hemi):
    from pymeg import atlas_glasser

    df = pd.read_hdf(
        "/home/nwilming/conf_analysis/results/all_contrasts_confmeg-20190516.hdf"
    )
    if epoch == "stimulus":
        time_cutoff = (-0.5, 1.35)
    else:
        time_cutoff = (-1, 0.5)
    query = 'epoch=="%s" & contrast=="%s" & %s(hemi=="avg")' % (
        epoch,
        contrast,
        {True: "~", False: ""}[hemi],
    )
    df = df.query(query)
    all_stats = {}
    for (name, area) in atlas_glasser.areas.items():
        task = contrast_tfr.get_tfr(df.query('cluster=="%s"' % area), time_cutoff)
        all_stats.update(contrast_tfr.par_stats(*task, n_jobs=1))
    return all_stats


def plot_2epoch_mosaics(
    df,
    stats=False,
    contrasts=["all", "choice", "confidence", "confidence_asym", "hand", "stimulus"],
):
    import pylab as plt
    import numpy as np
    from pymeg.contrast_tfr import plot_2epoch_mosaic

    # fmt: off
    remove = ["vfcPrimary", "vfcEarly", "vfcV3ab", "vfcIPS01",
              "vfcIPS23", "JWG_IPS_PCeS", "vfcFEF", "HCPMMP1_dlpfc",
              "vfcLO", "vfcTO", "vfcVO", "vfcPHC", "Insula",
              "HCPMMP1_frontal_inferior", "HCPMMP1_premotor", "JWG_M1"]
    # fmt: on
    clusters = df.index.get_level_values("cluster")
    clusters = np.array([c in remove for c in clusters])
    df = df.loc[~clusters, :]
    for contrast in contrasts:
        for hemi in [True, False]:
            plt.figure()
            query = 'contrast=="%s" & %s(hemi=="avg")' % (
                contrast,
                {True: "~", False: ""}[hemi],
            )
            d = df.query(query)
            plot_2epoch_mosaic(d, stats=stats, cmap="RdBu")
            title = "%s, %s" % (
                contrast,
                {True: "Lateralized", False: "Hemis avg."}[hemi],
            )
            plt.suptitle(title)
            plt.savefig(
                "/Users/nwilming/Desktop/tfr_average_2e_%s_lat%s.pdf"
                % (contrast, hemi),
                bbox_inches="tight",
            )


def plot_stream_figures(
    df,
    stats=False,
    contrasts=["all", "choice", "stimulus"],
    flip_cbar=False,
    suffix="",
    gs=None,
    aspect='auto',
    title_palette={}
):
    import matplotlib
    from pymeg import contrast_tfr_plots
    import pylab as plt
    conf = contrast_tfr_plots.PlotConfig(
        {"stimulus": (-0.35, 1.1), "response": (-0.35, 0.1)},  # Time windows for epochs
        [
            "all",
            "choice",
            "confidence",
            "confidence_asym",
            "hand",
            "stimulus",
        ],  # Contrast names
        stat_windows={"stimulus": (-0.5, 1.35), "response": (-1, 0.5)},
    )

    conf.configure_epoch(
        "stimulus",
        **{
            "xticks": [0,1],
            "xticklabels": ['0\n Stim.    \non  ', '1\nStim.    \noff   '],
            "yticks": [25, 50, 75, 100],
            "yticklabels": [25, 50, 75, 100],
            "xlabel": "",
            "ylabel": "Frequency [Hz]",
        },
    )
    conf.configure_epoch(
        "response",
        **{
            "xticks": [0],
            "xticklabels": ['0\n       Resp-\n      onse'],
            "yticks": [25, 50, 75, 100],
            "yticklabels": [25, 50, 75, 100],
            "xlabel": "",
            "ylabel": "Frequency [Hz]",
        },
    )
    for key, values in {
        "all": {"vmin": -50, "vmax": 50},
        "choice": {"vmin": -25, "vmax": 25},
        "confidence": {"vmin": -25, "vmax": 25},
        "confidence_asym": {"vmin": -25, "vmax": 25},
        "hand": {"vmin": -25, "vmax": 25},
        "stimulus": {"vmin": -25, "vmax": 25},
    }.items():
        conf.configure_contrast(key, **values)
        from collections import namedtuple

    Plot = namedtuple(
        "Plot", ["name", "cluster", "location", "annot_y", "annot_x"]
    )

    top, middle, bottom = slice(0, 2), slice(1, 3), slice(2, 4)
    # fmt: off
    layout = [
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

    if flip_cbar:
        cmap = "RdBu"
    else:
        cmap = "RdBu_r"            
    for i, contrast_name in enumerate(contrasts):        

        fig = contrast_tfr_plots.plot_tfr_selected_rois(
            contrast_name, df, layout, conf, cluster_correct=stats, 
            cmap=cmap,
            gs=gs,
            aspect=aspect,
            title_palette=title_palette
            )
    return fig
        #plt.suptitle(contrast_name)
        #plt.savefig("/Users/nwilming/Desktop/nsf_%s-%s.pdf" % (contrast_name, suffix))
        #plt.savefig("/Users/nwilming/Desktop/nsf_%s-%s.svg" % (contrast_name, suffix))

 
def plot_baseline_only_stream_figures(
    df,
    stats=False,
    contrasts=["all", "choice", "stimulus"],
    flip_cbar=False,
    suffix="",
    gs=None,
    aspect='auto',
    title_palette={}
):
    import matplotlib
    from pymeg import contrast_tfr_plots
    import pylab as plt
    conf = contrast_tfr_plots.PlotConfig(
        {"stimulus": (-0.35, 0.1), 'response':(-0.1, 0)},  # Time windows for epochs
        [
            "all",
            "choice",
            "confidence",
            "confidence_asym",
            "hand",
            "stimulus",
        ],  # Contrast names
        stat_windows={"stimulus": (-0.35, 0.1),'response':(-0.1, 0)},
    )

    conf.configure_epoch(
        "stimulus",
        **{
            "xticks": [-0.25,0],
            "xticklabels": ['-0.25\n         Time [s]\n         (Baseline)', '0'],
            "yticks": [25, 50, 75, 100],
            "yticklabels": [25, 50, 75, 100],
            "xlabel": "",
            "ylabel": "Frequency [Hz]",
        },
    )

    conf.configure_epoch(
        "response",
        **{
            "xticks": [0],
            "xticklabels": ['0\n       Resp-\n      onse'],
            "yticks": [25, 50, 75, 100],
            "yticklabels": [25, 50, 75, 100],
            "xlabel": "",
            "ylabel": "Frequency [Hz]",
        },
    )
    for key, values in {
        "all": {"vmin": -25, "vmax": 25},
        "choice": {"vmin": -25, "vmax": 25},
        "confidence": {"vmin": -25, "vmax": 25},
        "confidence_asym": {"vmin": -25, "vmax": 25},
        "hand": {"vmin": -25, "vmax": 25},
        "stimulus": {"vmin": -25, "vmax": 25},
    }.items():
        conf.configure_contrast(key, **values)
    
    from collections import namedtuple

    Plot = namedtuple(
        "Plot", ["name", "cluster", "location", "annot_y", "annot_x"]
    )

    top, middle, bottom = slice(0, 2), slice(1, 3), slice(2, 4)
    # fmt: off
    layout = [
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

    if flip_cbar:
        cmap = "RdBu"
    else:
        cmap = "RdBu_r"            
    for i, contrast_name in enumerate(contrasts):        

        fig = contrast_tfr_plots.plot_tfr_selected_rois(
            contrast_name, df, layout, conf, cluster_correct=stats, 
            cmap=cmap,
            gs=gs,
            aspect=aspect,
            title_palette=title_palette,
            ignore_response=True,
            axvlines=[-0.25],
            )
    return fig
        #plt.suptitle(contrast_name)
        #plt.savefig("/Users/nwilming/Desktop/nsf_%s-%s.pdf" % (contrast_name, suffix))
        #plt.savefig("/Users/nwilming/Desktop/nsf_%s-%s.svg" % (contrast_name, suffix))

 
# Ignore following for now


def plot_labels(
    data, areas, locations, gs, stats=True, minmax=(10, 20), tslice=slice(-0.25, 1.35)
):
    """
    Plot TFRS for a set of ROIs. At most 6 labels.
    """
    labels = rois.filter_cols(data.columns, areas)
    import pylab as plt

    # import seaborn as sns
    # colors = sns.color_palette('bright', len(labels))

    p = None
    maxrow = max([row for row, col in locations])
    maxcol = max([row for row, col in locations])

    for (row, col), area in zip(locations, labels):

        plt.subplot(gs[row, col])
        ex_tfr = get_tfr(data.query('est_key=="F"'), area, tslice=tslice)
        s = get_tfr_stack(data.query('est_key=="F"'), area, tslice=tslice)
        if stats:
            t, p, H0 = stats_test(s)
            p = p.reshape(t.shape)
        cbar = _plot_tfr(
            area,
            ex_tfr.columns.values,
            ex_tfr.index.values,
            s.mean(0),
            p,
            title_color="k",
            minmax=minmax[0],
        )
        if (row + 2, col + 1) == gs.get_geometry():
            pass
        else:
            cbar.remove()
        plt.xticks([])

        if col > 0:
            plt.yticks([])
        else:
            plt.ylabel("Freq")
        plt.subplot(gs[row + 1, col])
        ex_tfr = get_tfr(data.query('est_key=="LF"'), area, tslice=tslice)
        s = get_tfr_stack(data.query('est_key=="LF"'), area, tslice=tslice)
        if stats:
            t, p, H0 = stats_test(s)
            p = p.reshape(t.shape)
        cbar = _plot_tfr(
            area,
            ex_tfr.columns.values,
            ex_tfr.index.values,
            s.mean(0),
            p,
            title_color="k",
            minmax=minmax[1],
        )
        cbar.remove()
        # plt.xticks([0, 0.5, 1])
        if row == maxrow:
            plt.xlabel("time")

            # plt.xticks([tslice.start, 0, tslice.stop])
        else:
            plt.xticks([])
