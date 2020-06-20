#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Decoding analyses for conf_meg data.

4.  Signed confidence and choice decoder: Same as MIDC and MDDC but with
    confidence folded into the responses (-2, -1, 1, 2)
5.  Unsigned confidence decoder: Same as MIDC and MDDC but decode
    confidence only.
"""

import logging
import os
import numpy as np
import pandas as pd

from os.path import join

from functools import partial
from itertools import product

from sklearn import linear_model, discriminant_analysis, svm
from sklearn.model_selection import (
    cross_validate,
    cross_val_predict,
    RandomizedSearchCV,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import RandomOverSampler

from conf_analysis.behavior import metadata
from conf_analysis.meg import preprocessing


from joblib import Memory

if "TMPDIR" in os.environ.keys():
    memory = Memory(cachedir=os.environ["PYMEG_CACHE_DIR"])
    inpath = "/nfs/nwilming/MEG/sr_labeled/aggs/"
    oglinpath = "/nfs/nwilming/MEG/sr_labeled/aggs/ogl"
    outpath = "/nfs/nwilming/MEG/sr_decoding/"
elif "RRZ_LOCAL_TMPDIR" in os.environ.keys():
    tmpdir = os.environ["RRZ_LOCAL_TMPDIR"]
    outpath = "/work/faty014/MEG/sr_labeled/aggs/"
    outpath = "/work/faty014/MEG/sr_decoding/"
    memory = Memory(cachedir=tmpdir)
else:
    inpath = "/home/nwilming/conf_meg/sr_labeled/aggs"
    oglinpath = "/home/nwilming/conf_meg/sr_labeled/aggs/ogl"
    outpath = "/home/nwilming/conf_meg/sr_decoding"
    memory = Memory(cachedir=metadata.cachedir)

n_jobs = 1

n_trials = {
    1: {"stimulus": 1565, "response": 245},
    2: {"stimulus": 1852, "response": 1697},
    3: {"stimulus": 1725, "response": 27},
    4: {"stimulus": 1863, "response": 1807},
    5: {"stimulus": 1812, "response": 877},
    6: {"stimulus": 1128, "response": 113},
    7: {"stimulus": 1766, "response": 1644},
    8: {"stimulus": 1872, "response": 1303},
    9: {"stimulus": 1767, "response": 1104},
    10: {"stimulus": 1404, "response": 1209},
    11: {"stimulus": 1787, "response": 1595},
    12: {"stimulus": 1810, "response": 1664},
    13: {"stimulus": 1689, "response": 1620},
    14: {"stimulus": 1822, "response": 1526},
    15: {"stimulus": 1851, "response": 1764},
}


def decoders():
    return {
        #"SSD": ssd_decoder,
        # "SSD_delta_contrast": partial(ssd_decoder, target_value="delta_contrast"),
        #"SSD_acc_contrast": partial(ssd_decoder, target_value="acc_contrast"),
        #"SSD_varlow": partial(ssd_decoder, filter_var=0.05),
        #"SSD_varmed": partial(ssd_decoder, filter_var=0.1),
        #"SSD_varhigh": partial(ssd_decoder, filter_var=0.15),
        #"SSD_acc_contrast_varlow": partial(ssd_decoder, target_value="acc_contrast", filter_var=0.05),
        #"SSD_acc_contrast_varmed": partial(ssd_decoder, target_value="acc_contrast", filter_var=0.1),
        #"SSD_acc_contrast_varhigh": partial(ssd_decoder, target_value="acc_contrast", filter_var=0.15),
        # "SSD_acc_contrast_diff": partial(ssd_decoder, target_value="acc_contrast_diff"),
        # "MIDC_split": partial(midc_decoder, splitmc=True, target_col="response"),
        "MIDC_split_conf_split_mc": 
            partial(midc_decoder_split_confidence, splitmc=True, target_col="response"),
        "MIDC_split_conf": 
            partial(midc_decoder_split_confidence, splitmc=False, target_col="response"),
        # "MIDC_nosplit": partial(midc_decoder, splitmc=False, target_col="response"),
        # "SIDE_nosplit": partial(midc_decoder, splitmc=False, target_col="side"),
        # "CONF_signed": partial(
        #    midc_decoder, splitmc=False, target_col="signed_confidence"
        # ),
        #"CONF_signed_respsplit": partial(
        #    midc_decoder,
        #    splitmc=False,
        #    split_resp=True,
        #    target_col="unsigned_confidence",
        #),
        # "CONF_unsigned": partial(
        #    midc_decoder, splitmc=False, target_col="unsigned_confidence"
        # ),
        # "CONF_unsign_split": partial(
        #    midc_decoder, splitmc=True, target_col="unsigned_confidence"
        # ),
    }


def set_n_threads(n):
    import os

    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OMP_NUM_THREADS"] = str(n)


def submit(
    cluster="UKE",
    subjects=range(1, 16),
    ssd=False,
    epochs=["stimulus", "response"],
    ogl=False,
):
    from pymeg import parallel

    if ogl:
        hemis = ["Pair"]
    else:
        hemis = ["Pair", "Lateralized", "Averaged"]
    decoder = decoders().keys()
    if ssd:
        decoder = [d for d in decoder if "SSD" in d]
    else:
        decoder = [d for d in decoder if not ("SSD" in d)]
    if cluster == "SLURM":
        pmap = partial(
            parallel.pmap,
            email=None,
            tasks=1,
            nodes=1,
            memory=60,
            ssh_to=None,
            walltime="72:00:00",  # walltime='72:00:00',
            cluster="SLURM",
            env="py36",
        )
    else:
        pmap = partial(
            parallel.pmap,
            nodes=1,
            tasks=1,
            memory=61,
            ssh_to=None,
            walltime="172:00:00",
            env="py36",
        )

    for subject, epoch, dcd in product(subjects, epochs, decoder):
        filename = get_save_path(subject, dcd, epoch, ogl=ogl)
        import os.path

        if os.path.isfile(filename):
            print("Skipping ", subject, epoch, dcd)
            continue
        pmap(
            run_decoder,
            [(subject, dcd, epoch, hemis, ogl)],
            name="not_super_long" + dcd + epoch + str(subject),
        )


def get_save_path(subject, decoder, epoch, ogl=False):
    if ogl:
        filename = outpath + "/concatoglmis_S%i-%s-%s-decoding.hdf" % (
            subject,
            decoder,
            epoch,
        )
    else:
        filename = outpath + "/concat_S%i-%s-%s-decoding.hdf" % (
            subject,
            decoder,
            epoch,
        )
    return filename


def augment_meta(meta):
    meta.loc[:, "signed_confidence"] = (
        meta.loc[:, "confidence"] * meta.loc[:, "response"]
    ).astype(int)
    meta.loc[:, "unsigned_confidence"] = (meta.loc[:, "confidence"] == 1).astype(int)
    return meta


def run_decoder(
    subject,
    decoder,
    epoch,
    hemis=["Pair", "Lateralized", "Averaged"],
    ogl=False,
    ntasks=n_jobs,
):
    """
    Parallelize across areas and hemis.
    """
    set_n_threads(1)
    from multiprocessing import Pool
    from glob import glob

    from pymeg import aggregate_sr as asr
    from conf_analysis.meg import srtfr

    if ogl:
        clusters = srtfr.get_ogl_clusters()
        filenames = glob(
            join(oglinpath, "missing_ogl_S%i_*_%s_agg.hdf" % (subject, epoch))
        )
    else:
        clusters = srtfr.get_clusters()
        filenames = glob(join(inpath, "S%i_*_%s_agg.hdf" % (subject, epoch)))
    print(filenames)
    areas = clusters.keys()
    areas = [x for x in areas if not x.startswith("NSWFRONT")]
    print("Areas:", areas)
    meta = augment_meta(preprocessing.get_meta_for_subject(subject, "stimulus"))
    # meta = meta.dropna(subset=['contrast_probe'])
    args = []
    for area, hemi in product(areas, hemis):
        if hemi == "Pair":
            area = [area + "_RH", area + "_LH"]
        a = (meta, asr.delayed_agg(filenames, hemi=hemi, cluster=area), decoder, hemi)
        args.append(a)
    print("Processing %i tasks" % (len(args)))
    from contextlib import closing

    scores = []
    # for arglist in range(0, len(args), ntasks):
    with closing(Pool(ntasks, maxtasksperchild=1)) as p:
        sc = p.starmap(
            _par_apply_decode, args  # ,[arglist : arglist + ntasks]
        )  # , chunksize=ntasks)
        sc = [s for s in sc if s is not None]
    scores.extend(sc)
    print("Concat ", len(scores))
    scores = pd.concat(scores)
    filename = get_save_path(subject, decoder, epoch, ogl=ogl)
    # outpath + "/concatogl_S%i-%s-%s-decoding.hdf" % (subject, decoder, epoch)
    scores.loc[:, "subject"] = subject
    print("Saving as ", filename)
    scores.to_hdf(filename, "decoding")
    p.terminate()
    return scores


def _par_apply_decode(meta, delayed_agg, decoder, hemi=None):
    print(delayed_agg)
    agg = delayed_agg()
    dt = np.diff(agg.columns.get_level_values("time"))[0]
    latencies = None
    if "SSD" in decoder:
        latencies = np.arange(-0.1, 0.65, dt)
    # print('Starting', delayed_agg)
    scores = apply_decoder(
        meta, agg, decoders()[decoder], latencies=latencies, hemi=hemi
    )
    print("Ending", scores.memory_usage().sum() / 1024 / 1024)
    return scores


def apply_decoder(meta, agg, decoder, latencies=None, hemi=None):
    """Run a decoder across a set of latencies.

    Args:
        agg: pd.DataFrame
            Aggregate data frame.
        decoder: function
            Function that takes meta data, agg data, areas
            and latency as input and returns decoding scores
            as data frame.
        latencies: None or list
            List of latencies to decode across. If None
            all time points in agg will be used.
    """
    import time

    # Kick out trials that have choice_rt < 0.225
    meta = meta.set_index("hash")
    choice_rt = meta.choice_rt
    valid_trials = choice_rt[choice_rt >= 0.225].index.values
    trial_id = agg.index.get_level_values("trial")
    agg = agg.loc[np.isin(trial_id, valid_trials)]
    meta = meta.reset_index()
    # How many kicked out?
    n_out = np.isin(
        trial_id.unique(), valid_trials, invert=True, assume_unique=True
    ).sum()
    n_all = len(trial_id.unique())
    print(
        "Kicking out %i/%i (%0.2f percent) trials due to RT"
        % (n_out, n_all, n_out / n_all)
    )

    start = time.time()
    if latencies is None:
        latencies = agg.columns.get_level_values("time").values

    area = np.unique(agg.index.get_level_values("cluster"))
    scores = []
    for latency in latencies:
        # logging.info("Applying decoder %s at latency %s" % (decoder, latency))
        try:
            s = decoder(meta, agg, area, latency=latency)
            s.loc[:, "cluster"] = str(area)
            scores.append(s)
        except:
            logging.exception(
                """'Error in run_decoder:        
        # Decoder: %s
        # Area: %s
        # Latency: %f)"""
                % (str(decoder), area, latency)
            )
            raise
    res = pd.concat(scores)
    res.loc[:, "hemi"] = hemi
    logging.info(
        "Applying decoder %s across N=%i latencies took %3.2fs"
        % (decoder, len(latencies), time.time() - start)
    )
    return res


def ssd_decoder(meta, data, area, latency=0.18, target_value="contrast", filter_var=None):
    """
    Sensory signal decoder (SSD).

    Each frequency and brain area is one feature, contrast is the target.
    Args:
        meta: DataFrame
            Metadata frame that contains meta data per row
        data: Aggregate data frame
        area: List or str
            Which areas to use for decoding. Multiple areas provide
            independent features per observation
        latency: float
            Which time point to decode
        target_value: str
            Which target to decode (refers to a col. in meta)

    """

    from sklearn.metrics import make_scorer
    from scipy.stats import linregress

    slope_scorer = make_scorer(lambda x, y: linregress(x, y)[0])
    corr_scorer = make_scorer(lambda x, y: np.corrcoef(x, y)[0, 1])
    meta = meta.set_index("hash")
    sample_scores = []
    if filter_var is not None:
        meta = meta.query('noise_sigma==%f'%filter_var)
        data = data.loc[np.isin(data.index.get_level_values('trial'), meta.index)]

    for sample_num, sample in enumerate(np.arange(0, 1, 0.1)):
        # Map sample times to existing time points in data
        target_time_point = sample + latency
        times = data.columns.get_level_values("time").values
        target_time_point = times[np.argmin(abs(times - target_time_point))]

        # Compute mean latency
        latency = target_time_point - sample

        # Turn data into (trial X Frequency)
        X = []
        for a in ensure_iter(area):
            x = pd.pivot_table(
                data.query('cluster=="%s"' % a),
                index="trial",
                columns="freq",
                values=target_time_point,
            )
            X.append(x)
        sample_data = pd.concat(X, 1)

        # Build target vector
        cvals = np.stack(meta.loc[sample_data.index, "contrast_probe"].values)
        if target_value == "contrast":
            target = cvals[:, sample_num]

        elif target_value == "delta_contrast":
            if sample_num == 0:
                target = cvals[:, sample_num]
            else:
                target = cvals[:, sample_num] - cvals[:, sample_num - 1]
        elif target_value == "acc_contrast":
            target = cvals[:, : (sample_num + 1)].mean(1)
        elif target_value == "acc_contrast_diff":
            if sample_num == 0:
                target = cvals[:, sample_num]
            else:
                target = cvals[:, sample_num] - cvals[:, :(sample_num)].mean(1)
        else:
            raise RuntimeError("Do not understand target: %s" % target)

        metrics = {
            "explained_variance": "explained_variance",
            "r2": "r2",
            "slope": slope_scorer,
            "corr": corr_scorer,
        }

        classifiers = {
            # "OLS": linear_model.LinearRegression(),
            "Ridge": linear_model.RidgeCV()
        }

        for name, clf in list(classifiers.items()):
            clf = Pipeline([("Scaling", StandardScaler()), (name, clf)])
            score = cross_validate(
                clf,
                sample_data.values,
                target,
                cv=10,
                scoring=metrics,
                return_train_score=False,
                n_jobs=1,
            )  # n_jobs = 1 because it
            # is nested par loop
            # fit = clf(sample_data.values, target)
            del score["fit_time"]
            del score["score_time"]
            score = {k: np.mean(v) for k, v in list(score.items())}
            score["latency"] = latency
            score["Classifier"] = name
            score["sample"] = sample_num
            if filter_var is not None:
                score['noise_sigma'] = filter_var
            # score['coefs'] = fit.coef_.astype(object)
            sample_scores.append(score)
        del sample_data

    return pd.DataFrame(sample_scores)


def midc_decoder(
    meta,
    data,
    area,
    latency=0,
    splitmc=True,
    split_resp=False,
    target_col="response",
    predict=False,
):
    """
    This signal decodes choices of participants based on local
    reconstructed brain activity. The decoding needs to be decoupled
    from evidence strength. To this end decoding will be carried out
    in several bins that all have the same mean contrast.

    This will be somewhat challenging because for
    high mean contrast behavior will obviously be very different, so
    very uneven class sizes. I'd use overlapping bins of equal size here.
    An interesting question would be to see if the decoder transfers
    between bins, this would be the strongest test of evidence
    independent decoding.

    Set the input data frame and area to 'Havg' for motor independent
    signals, and to 'Lateralized' for motor dependent choice signals
    (MDDC).
    """
    meta = meta.set_index("hash")
    # Map to nearest time point in data
    times = data.columns.get_level_values("time").values
    target_time_point = times[np.argmin(abs(times - latency))]
    data = data.loc[:, target_time_point]

    # Turn data into (trial X Frequency)
    X = []
    for a in ensure_iter(area):
        x = pd.pivot_table(
            data.query('cluster=="%s"' % a),
            index="trial",
            columns="freq",
            values=target_time_point,
        )
        X.append(x)
    data = pd.concat(X, 1)
    meta = meta.loc[data.index, :]
    scores = []
    if splitmc or split_resp:
        if splitmc:
            selector = meta.mc < 0.5
            split_label = "mc<0.5"
        elif split_resp:
            selector = meta.response == 1
            split_label = 'R1'
        for mc, sub_meta in meta.groupby(selector):
            sub_data = data.loc[sub_meta.index, :]
            sub_meta = sub_meta.loc[sub_data.index, :]
            # Buld target vector
            target = (sub_meta.loc[sub_data.index, target_col]).astype(int)
            if not predict:
                score = categorize(target, sub_data, target_time_point, predict=False)
                score.loc[:, split_label] = mc
            else:
                score = categorize(target, sub_data, target_time_point, predict=True)
                # score = score["SCVlin"]
                score = pd.DataFrame(score, index=sub_data.index)
            scores.append(score)
        scores = pd.concat(scores)
    else:
        # Buld target vector
        target = (meta.loc[data.index, target_col]).astype(int)
        if not predict:
            scores = categorize(target, data, target_time_point, predict=False)
        else:
            scores = categorize(target, data, target_time_point, predict=True)
            # scores = scores["SCVlin"]
            scores = pd.DataFrame(scores, index=data.index)
    scores.loc[:, "area"] = str(area)
    scores.loc[:, "latency"] = latency
    return scores



def midc_decoder_split_confidence(
    meta,
    data,
    area,
    latency=0,
    splitmc=True,
    split_confidence=True,
    target_col="response",
    predict=False,
):
    """
    Copy of MIDC Decoder - Split by confidence and see if response can be decoded.

    Copying this to not mess with the original midc decoder. Ugly.
    """
    meta = meta.set_index("hash")
    # Map to nearest time point in data
    times = data.columns.get_level_values("time").values
    target_time_point = times[np.argmin(abs(times - latency))]
    data = data.loc[:, target_time_point]

    # Turn data into (trial X Frequency)
    X = []
    for a in ensure_iter(area):
        x = pd.pivot_table(
            data.query('cluster=="%s"' % a),
            index="trial",
            columns="freq",
            values=target_time_point,
        )
        X.append(x)
    data = pd.concat(X, 1)
    meta = meta.loc[data.index, :]
    scores = []

    selector = []
    splitter = []
    if splitmc:
        selector.append(meta.mc < 0.5)            
        splitter.append('mc')
    if split_confidence:
        selector.append(meta.confidence == 1)            
        splitter.append('conf')
    for mc, sub_meta in meta.groupby(selector):
        sub_data = data.loc[sub_meta.index, :]
        sub_meta = sub_meta.loc[sub_data.index, :]
        # Buld target vector
        target = (sub_meta.loc[sub_data.index, target_col]).astype(int)
        if not predict:
            score = categorize(target, sub_data, target_time_point, predict=False)
            score.loc[:, "split_label"] = str(splitter)+str(mc)
            print(str(splitter)+str(mc))
        else:
            score = categorize(target, sub_data, target_time_point, predict=True)
            # score = score["SCVlin"]
            score = pd.DataFrame(score, index=sub_data.index)
        scores.append(score)
    scores = pd.concat(scores)
   
    scores.loc[:, "area"] = str(area)
    scores.loc[:, "latency"] = latency
    return scores


def multiclass_roc(y_true, y_predict, **kwargs):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(
        label_binarize(y_true, classes=[-2, -1, 1, 2]),
        label_binarize(y_predict, classes=[-2, -1, 1, 2]),
        **kwargs
    )


def categorize(target, data, latency, predict=False):
    """
    Expects a pandas series and a pandas data frame.
    Both need to be indexed with the same index.
    """
    from imblearn.pipeline import Pipeline
    from sklearn.metrics.scorer import make_scorer
    from sklearn.metrics import recall_score, precision_score
    from sklearn.utils.multiclass import type_of_target

    if not all(target.index.values == data.index.values):
        raise RuntimeError("Target and data not aligned with same index.")

    target = target.values
    data = data.values
    # Determine prediction target:
    y_type = type_of_target(target)
    if y_type == "multiclass":
        metrics = {
            "roc_auc": make_scorer(multiclass_roc, average="weighted"),
            "accuracy": "accuracy",
        }
    else:
        metrics = ["roc_auc", "accuracy", "precision", "recall"]
    classifiers = {"SCVlin": svm.SVC(kernel="linear", probability=True)}

    scores = []
    for name, clf in list(classifiers.items()):
        clf = Pipeline(
            [
                ("Scaling", StandardScaler()),
                # ('PCA', PCA(n_components=20)),
                ("Upsampler", RandomOverSampler(sampling_strategy="minority")),
                (name, clf),
            ]
        )
        if not predict:
            score = cross_validate(
                clf,
                data,
                target,
                cv=10,
                scoring=metrics,
                return_train_score=False,
                n_jobs=1,
            )
            del score["fit_time"]
            del score["score_time"]
            score = {k: np.mean(v) for k, v in list(score.items())}
            score["latency"] = latency
            score["Classifier"] = name
            scores.append(score)
        else:
            scores = cross_val_predict(
                clf, data, target, cv=10, method="predict_proba", n_jobs=1
            )
    if not predict:
        return pd.DataFrame(scores)
    else:
        return scores


def SVMCV(params):
    return RandomizedSearchCV(svm.SVC(), params, n_iter=50, n_jobs=4)


def submit_cross_area_decoding():
    from pymeg import parallel

    pmap = partial(
        parallel.pmap,
        email=None,
        tasks=1,
        nodes=1,
        memory=60,
        ssh_to=None,
        walltime="72:00:00",  # walltime='72:00:00',
        cluster="SLURM",
        env="py36",
    )

    for subject in range(1, 16):
        for odd in [True, False]:
            pmap(run_cross_area_decoding, [(subject, [odd])], name="XA" + str(subject))


def run_cross_area_decoding(subject, odd_partition=[True, False], ntasks=15):
    set_n_threads(1)
    from multiprocessing import Pool
    from glob import glob
    from pymeg import atlas_glasser, aggregate_sr as asr
    import os
    from os.path import join
    from glob import glob

    low_level_areas = [
        "vfcPrimary",
        "vfcEarly",
        "vfcV3ab",
        "vfcIPS01",
        "vfcIPS23",
        "vfcLO",
        "vfcTO",
        "vfcVO",
        "vfcPHC",
        "JWG_IPS_PCeS",
        "vfcFEF",
        "HCPMMP1_dlpfc",
        "HCPMMP1_insular_front_opercular",
        "HCPMMP1_frontal_inferior",
        "HCPMMP1_premotor",
        "JWG_M1",
    ]
    high_level_areas = [
        "JWG_IPS_PCeS",
        "vfcFEF",
        "HCPMMP1_dlpfc",
        "HCPMMP1_premotor",
        "JWG_M1",
    ]
    # First load low level averaged stimulus data
    filenames = glob(join(inpath, "S%i_*_%s_agg.hdf" % (subject, "stimulus")))

    args = []
    filename = outpath + "/cross_area_S%i-decoding_%s.hdf" % (
        subject,
        str(odd_partition),
    )
    print("Save target:", filename)
    cnt = 0
    add_meta = {"subject": subject, "low_hemi": "Averaged", "high_hemi": "Lateralized"}
    for lla, hla in product(low_level_areas, high_level_areas):
        for odd in odd_partition:
            low_level_data = asr.delayed_agg(filenames, hemi="Averaged", cluster=lla)
            high_level_data = asr.delayed_agg(
                filenames, hemi="Lateralized", cluster=hla
            )
            args.append(
                (
                    subject,
                    low_level_data,
                    lla,
                    high_level_data,
                    hla,
                    0.18,
                    "response",
                    "cross_dcd_S%i-" % subject,
                    "%s" % cnt,
                    add_meta,
                    odd,
                )
            )
            cnt += 1
    print("There are %i decoding tasks for subject %i" % (len(args), subject))
    scratch = os.environ["TMPDIR"]
    try:
        with Pool(ntasks, maxtasksperchild=1) as p:
            scores = p.starmap(motor_decoder, args)  # , chunksize=ntasks)
    finally:
        # Do this to find out why memory error occurs
        print("Saving data")
        try:
            cmd = join(scratch, "*.hdf")
            os.system("cp %s /nfs/nwilming/MEG/scratch/" % join(scratch, "*.hdf"))
        except Exception:
            pass

        # Now collect all data from scratch
        save_filenames = join(scratch, "cross_dcd" + "*.hdf")
        scores = [pd.read_hdf(x) for x in glob(save_filenames)]
        scores = pd.concat(scores)
        # scores.loc[:, 'subject'] = subject
        scores.to_hdf(filename, "decoding")
        return scores


def motor_decoder(
    subject,
    low_level_data,
    low_level_area,
    motor_data,
    motor_area,
    low_level_peak,
    target_col="response",
    save_filename=None,
    save_prefix=None,
    add_meta={},
    odd_times=None,
):
    """
    This signal predicts motor activity from a set of time points in early visual cortex.

    The idea is to train to decoders simultaneously:

    1) Train a motor decoder and predict log(probability) of a choice. Do this 
       on the training set.
    2) Train a decoder from early visual cortex data that predicts the 
       log(probability) of a choice. Train on training set, evaluate on test.
    """
    meta = augment_meta(preprocessing.get_meta_for_subject(subject, "stimulus"))
    meta = meta.set_index("hash")
    low_level_data = low_level_data()
    motor_data = motor_data()

    # Iterate over all high level time points:
    times = motor_data.columns.get_level_values("time").values
    dt = np.diff(times)[0]
    t_idx = (-0.4 < times) & (times < 0.1)

    cnt = 0
    time_points = times[t_idx]
    if odd_times is not None:
        if odd_times:
            time_points = time_points[:-1:2]
        else:
            time_points = time_points[1::2]
    for high_level_latency in time_points:
        print("High level latency:", high_level_latency)
        target_time_point = times[np.argmin(abs(times - high_level_latency))]
        md = prep_motor_data(motor_area, motor_data, target_time_point)

        # Buld target vector
        motor_target = (meta.loc[md.index, target_col]).astype(int)
        all_scores = []
        for low_level_latency in np.arange(-0.1, 0.2 + dt, dt):
            lld_data = prep_low_level_data(
                low_level_area, low_level_data, low_level_peak, low_level_latency
            )
            # low_level_meta = meta.loc[lld_data.index, :]
            assert all(lld_data.index == motor_target.index)
            scores = chained_categorize(motor_target, md, lld_data)
            scores.loc[:, "low_level_latency"] = low_level_latency
            all_scores.append(scores)
        save_all_scores(
            all_scores,
            add_meta,
            save_filename,
            save_prefix,
            cnt,
            high_level_latency,
            low_level_peak,
            low_level_area,
            motor_area,
        )
        cnt += 1
        del all_scores
        del md
        del motor_target
    del motor_data
    del low_level_data


def prep_motor_data(motor_area, motor_data, target_time_point):
    md = motor_data.loc[:, target_time_point]
    # Turn data into (trial X Frequency)
    X = []
    for a in ensure_iter(motor_area):
        x = pd.pivot_table(
            md.query('cluster=="%s"' % a),
            index="trial",
            columns="freq",
            values=target_time_point,
        )
        X.append(x)
    return pd.concat(X, 1)


def prep_low_level_data(
    low_level_area, low_level_data, low_level_peak, low_level_latency
):
    lld = []
    low_times = low_level_data.columns.get_level_values("time").values
    for s in np.arange(0, 1, 0.1) + low_level_peak + low_level_latency:
        # Turn data into (trial X Frequency)
        low_target_time_point = low_times[np.argmin(abs(low_times - s))]
        for a in ensure_iter(low_level_area):
            x = pd.pivot_table(
                low_level_data.query('cluster=="%s"' % a),
                index="trial",
                columns="freq",
                values=low_target_time_point,
            )
            lld.append(x)
    return pd.concat(lld, 1)


def save_all_scores(
    all_scores,
    add_meta,
    save_filename,
    save_prefix,
    cnt,
    high_level_latency,
    low_level_peak,
    low_level_area,
    motor_area,
):
    all_scores = pd.concat(all_scores)
    all_scores.loc[:, "high_level_latency"] = high_level_latency
    all_scores.loc[:, "low_level_peak"] = low_level_peak
    all_scores.loc[:, "low_level_area"] = low_level_area
    all_scores.loc[:, "high_level_area"] = motor_area
    for key, value in add_meta.items():
        all_scores.loc[:, key] = value
    scratch = os.environ["TMPDIR"]
    sf = join(scratch, save_filename + save_prefix + "_%i.hdf" % cnt)
    print("Saving to ", sf)
    all_scores.to_hdf(sf, "df")


def chained_categorize(target_a, data_a, data_b):
    """
    Trains a classifier to predict target_a from data_a. Then
    predicts log(pA) with that for training set and trains a 
    second classifier to predict log(pA) with data_b.

    target_a, data_a and data_b all need to have the same index.
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import roc_auc_score, mean_squared_error
    from sklearn.decomposition import PCA
    from sklearn.utils import shuffle
    from imblearn.pipeline import Pipeline

    if not (
        all(target_a.index.values == data_a.index.values)
        and all(data_a.index.values == data_b.index.values)
    ):
        raise RuntimeError("Target and data not aligned with same index.")

    target_a = target_a.values
    data_a = data_a.values
    data_b = data_b.values
    # Determine prediction target:
    # y_type = type_of_target(target_a)

    metrics = ["roc_auc", "accuracy"]
    classifier_a = svm.SVC(kernel="linear", probability=True)
    classifier_a = Pipeline(
        [
            ("Scaling", StandardScaler()),
            ("Upsampler", RandomOverSampler(sampling_strategy="minority")),
            ("SVM", classifier_a),
        ]
    )

    classifier_b = Pipeline(
        [
            ("Scaling", StandardScaler()),
            ("PCA", PCA(n_components=0.95, svd_solver="full")),
            ("linear_regression", LinearRegression()),
        ]
    )
    classifier_b_baseline = Pipeline(
        [
            ("Scaling", StandardScaler()),
            ("PCA", PCA(n_components=0.95, svd_solver="full")),
            ("linear_regression", LinearRegression()),
        ]
    )

    scores = []
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for train_index, test_index in sss.split(data_a, target_a):
        train_data_a = data_a[train_index]
        train_target_a = target_a[train_index]
        clf_a = classifier_a.fit(train_data_a, train_target_a)

        train_data_b = data_b[train_index]
        target_b = clf_a.predict_log_proba(train_data_a)[:, 0]
        clf_b = classifier_b.fit(train_data_b, target_b)
        clf_b_baseline = classifier_b_baseline.fit(train_data_b, shuffle(target_b))
        classifier_a_roc = roc_auc_score(
            target_a[test_index], clf_a.predict_log_proba(data_a[test_index])[:, 0]
        )

        clf_a_test_predicted = clf_a.predict_log_proba(data_a[test_index])[:, 0]
        clf_b_test_predicted = clf_b.predict(data_b[test_index])
        clf_b_baseline_test_predicted = clf_b_baseline.predict(data_b[test_index])

        classifier_b_msqerr = mean_squared_error(
            clf_a_test_predicted, clf_b_test_predicted
        )
        classifier_b_shuffled_msqerr = mean_squared_error(
            clf_a_test_predicted, clf_b_baseline_test_predicted
        )

        classifier_b_corr = np.corrcoef(clf_a_test_predicted, clf_b_test_predicted)[
            0, 1
        ]
        classifier_b_shuffled_corr = np.corrcoef(
            clf_a_test_predicted, clf_b_baseline_test_predicted
        )[0, 1]

        scores.append(
            {
                "classifier_a_roc": classifier_a_roc,
                "classifier_b_msqerr": classifier_b_msqerr,
                "classifier_b_shuffled_msqerr": classifier_b_shuffled_msqerr,
                "classifier_b_corr": classifier_b_corr,
                "classifier_b_shuffled_corr": classifier_b_shuffled_corr,
                # "classifier_b_weights": clf_b.steps[-1][1].coef_,
            }
        )

    return pd.DataFrame(scores)


def get_tableau(
    meta,
    dresp,
    areas={"lh": "M1-lh", "rh": "M1-rh"},
    field="response",
    options=[1, 1, -1, -1],
    dbase=None,
    late=True,
    **kwargs
):
    """
    Computes the response lateralization 'Tableau', i.e. Hemishphere*response
    plot.s
    """
    rois = list(areas.values()) + ["time", "trial", "est_key", "est_val"]
    import pylab as plt

    # dresp = dresp.reset_index().loc[:, rois].query('est_key=="F"')
    # dresp.set_index(['trial', 'time', 'est_key', 'est_val'], inplace=True)
    dresp = dresp.reset_index().set_index("trial")
    plt.clf()

    meta = meta.loc[np.unique(dresp.index.values), :]
    for i, (resp, area) in enumerate(
        zip(options, [areas["lh"], areas["rh"], areas["lh"], areas["rh"]])
    ):
        plt.subplot(2, 2, i + 1)

        index = meta.query("%s==%i" % (field, resp)).index
        data = dresp.loc[index, :]

        k = pd.pivot_table(data, values=area, index="est_val", columns="time")

        dtmin, dtmax = k.columns.min(), k.columns.max()
        dfmin, dfmax = k.index.min(), k.index.max()
        plt.imshow(
            np.flipud(k),
            cmap="RdBu_r",
            aspect="auto",
            extent=[dtmin, dtmax, dfmin, dfmax],
            **kwargs
        )
        plt.text(0, 20, "%s:%i, HEMI=%s" % (field, resp, area))
        if i == 0:
            plt.title("LH")
            plt.xlabel("%s=%i" % (field, resp))
        if i == 1:
            plt.title("RH")


def get_path(epoch, subject, session, cache=False):
    from os.path import join
    from glob import glob

    if not cache:
        path = metadata.sr_labeled
        if epoch == "stimulus":
            filenames = glob(
                join(path, "S%i-SESS%i-stimulus-F-chunk*-lcmv.hdf" % (subject, session))
            )
        elif epoch == "response":
            filenames = glob(
                join(path, "S%i-SESS%i-response-F-chunk*-lcmv.hdf" % (subject, session))
            )
        else:
            raise RuntimeError("Do not understand epoch %s" % epoch)
    else:
        path = metadata.sraggregates
        if epoch == "stimulus":
            filenames = join(path, "S%i-SESS%i-stimulus-lcmv.hdf" % (subject, session))
        elif epoch == "response":
            filenames = join(path, "S%i-SESS%i-response-lcmv.hdf" % (subject, session))
    return filenames


def submit_aggregates(cluster="uke", only_glasser=False):
    from pymeg import parallel
    import time

    for subject, epoch, session in product(
        [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 3, 9], ["stimulus"], range(4)
    ):
        parallel.pmap(
            aggregate,
            [(subject, session, epoch, only_glasser)],
            name="agg" + str(session) + epoch + str(subject),
            tasks=8,
            memory=60,
            walltime="12:00:00",
        )


def aggregate(subject, session, epoch, glasser=False):
    """
    Glasser implies aggregating only the glasser ROIs
    """
    from conf_analysis.meg import srtfr
    from pymeg import aggregate_sr as asr, atlas_glasser as ag
    from os.path import join

    if not glasser:
        stim = (
            "/home/nwilming/conf_meg/sr_labeled/S%i-SESS%i-stimulus-*-chunk*-lcmv.hdf"
            % (subject, session)
        )
        resp = (
            "/home/nwilming/conf_meg/sr_labeled/S%i-SESS%i-response-*-chunk*-lcmv.hdf"
            % (subject, session)
        )
        all_clusters = srtfr.get_clusters()
        filename = join(
            "/home/nwilming/conf_meg/sr_labeled/aggs/",
            "S%i_SESS%i_%s_agg.hdf" % (subject, session, epoch),
        )
    else:
        stim = (
            "/home/nwilming/conf_meg/sr_labeled/ogl/S%i-SESS%i-stimulus-*-chunk*-lcmv.hdf"
            % (subject, session)
        )
        resp = (
            "/home/nwilming/conf_meg/sr_labeled/ogl/S%i-SESS%i-response-*-chunk*-lcmv.hdf"
            % (subject, session)
        )
        # all_clusters = ag.get_all_glasser_clusters()
        all_clusters = srtfr.get_ogl_clusters()
        # all_clusters = {'5L': all_clusters['5L'], 'PSL':all_clusters['PSL'], 'SFL':all_clusters['SFL']}
        filename = join(
            "/home/nwilming/conf_meg/sr_labeled/aggs/ogl/",
            "missing_ogl_S%i_SESS%i_%s_agg.hdf" % (subject, session, epoch),
        )
    print("Will save agg as:", filename)
    if epoch == "stimulus":
        agg = asr.aggregate_files(stim, stim, (-0.25, 0), all_clusters=all_clusters)
    elif epoch == "response":
        agg = asr.aggregate_files(resp, stim, (-0.25, 0), all_clusters=all_clusters)
    asr.agg2hdf(agg, filename)


def slimdown_aggs(source):
    from glob import glob
    from os.path import join, split

    sources = glob(join(source, "ogl*.hdf"))
    for source in sources:
        path, fname = split(source)
        dest = join(path, "slimdown", fname)
        print(source, " -> ", dest)
        slimdown_agg(source, dest, "Pair")


def slimdown_agg(source, dest, group, even=True):
    import h5py

    with h5py.File(source, "r") as s, h5py.File(dest, "w") as d:
        paths = get_all_paths(s[group])
        for p in paths:
            # print('Copying ', p)
            hdf_copydset(s, d, s[p], even=even)


def get_all_paths(source):
    if "ndim" in dir(source):
        return [source.name]
    else:
        paths = []
        for value in source.values():
            paths += get_all_paths(value)
        return paths


def hdf_copydset(source, dest, dataset, even=True):
    try:
        group = dest[dataset.parent.name]
    except KeyError:
        group = dest.create_group(dataset.parent.name)
    if even:
        idx = slice(None, None, 2)
    else:
        idx = slice(1, None, 2)
    ds = group.create_dataset(dataset.name, data=dataset[:, idx])

    for key, item in dataset.attrs.items():
        if key.startswith("col"):
            ds.attrs[key] = item[idx]
        else:
            ds.attrs[key] = item


def get_all_ortho_IPS():
    scores = []
    for subject in range(1, 16):
        scores.append(get_ortho_midc(subject))
    return pd.concat(scores)


def get_ortho_midc(subject):
    """
    Compute orthogonalized decoding values for IPS/PostCeS.
    """
    from pymeg import aggregate_sr as asr
    from glob import glob
    inpath = "/home/nwilming/conf_meg/sr_labeled/aggs"
    cluster = "JWG_M1"
    # First load low level averaged stimulus data
    filenames = glob(join(inpath, "S%i_*_%s_agg.hdf" % (subject, "stimulus")))
    data = asr.delayed_agg(
        filenames, hemi="Lateralized", cluster=[cluster] + ["JWG_IPS_PCeS"]
    )()
    meta = augment_meta(preprocessing.get_meta_for_subject(subject, "stimulus"))
    scores = ortho_midc_decoder(meta, data, "JWG_IPS_PCeS", "JWG_M1", latency=1.1)
    scores.loc[:, "subject"] = subject
    return scores


def get_ortho_x(data, area, ortho_area, latency):
    from scipy.stats import linregress
    times = data.columns.get_level_values("time").values
    target_time_point = times[np.argmin(abs(times - latency))]
    data = data.loc[:, target_time_point]

    # Turn data into (trial X Frequency)
    X = []
    for a in ensure_iter(area):
        x = pd.pivot_table(
            data.query('cluster=="%s"' % a),
            index="trial",
            columns="freq",
            values=target_time_point,
        )
        X.append(x)

        # Turn data into (trial X Frequency)
    O = []
    for a in ensure_iter(ortho_area):
        x = pd.pivot_table(
            data.query('cluster=="%s"' % a),
            index="trial",
            columns="freq",
            values=target_time_point,
        )
        O.append(x)
    ortho_data = pd.concat(O, 1)
    data = pd.concat(X, 1)

    for col in data.columns:
        x = ortho_data.loc[:, col].values
        y = data.loc[:, col].values
        slope, i, _, _, _ = linregress(x, y)
        data.loc[:, col] = y - slope * x + i
    return data, ortho_data


def ortho_midc_decoder(
    meta,
    data,
    area,
    ortho_area,
    latency=0,
    splitmc=True,
    target_col="response",
    predict=False,
):
    meta = meta.set_index("hash")
    # Map to nearest time point in data
    times = data.columns.get_level_values("time").values
    target_time_point = times[np.argmin(abs(times - latency))]
    data = data.loc[:, target_time_point]

    # Turn data into (trial X Frequency)
    data, ortho = get_ortho_x(data, area, ortho_area, latency)

    meta = meta.loc[data.index, :]
    scores = []
    if splitmc:
        for mc, sub_meta in meta.groupby(meta.mc < 0.5):
            sub_data = data.loc[sub_meta.index, :]
            sub_meta = sub_meta.loc[sub_data.index, :]
            # Buld target vector
            target = (sub_meta.loc[sub_data.index, target_col]).astype(int)
            if not predict:
                score = categorize(target, sub_data, target_time_point, predict=False)
                score.loc[:, "mc<0.5"] = mc
            else:
                score = categorize(target, sub_data, target_time_point, predict=True)
                # score = score["SCVlin"]
                score = pd.DataFrame(score, index=sub_data.index)
            scores.append(score)
        scores = pd.concat(scores)
    else:
        # Buld target vector
        target = (meta.loc[data.index, target_col]).astype(int)
        if not predict:
            scores = categorize(target, data, target_time_point, predict=False)
        else:
            scores = categorize(target, data, target_time_point, predict=True)
            # scores = scores["SCVlin"]
            scores = pd.DataFrame(scores, index=data.index)
    scores.loc[:, "area"] = str(area)
    scores.loc[:, "latency"] = latency
    return scores



@memory.cache()
def build_auc_design_matrix(
    subject, cluster, latency, hemi="Lateralized", freq_bands=None, ogl=False
):
    from glob import glob
    from pymeg import aggregate_sr as asr
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

    data = pd.pivot_table(data, values=latency, index='trial', columns='freq')
    if freq_bands is not None:
        data = data.loc[:, freq_bands[0]:freq_bands[1]].mean(1)
    trial_index = data.index.get_level_values("trial").unique()

    meta = preprocessing.get_meta_for_subject(subject, "stimulus")
    meta.set_index("hash", inplace=True)
    meta = meta.loc[trial_index]    
    return data, meta


def get_all_sanity_auc():
    AUCS = []
    for subject in range(1, 16):
        for cluster in ['JWG_M1', 'JWG_IPS_PCeS']:
            print(cluster)
            data, meta = build_auc_design_matrix(subject, cluster, 1.1, freq_bands=[15, 35])
            auc = sanity_check_auc(meta, data)
            if subject>=8:
                auc = 1-auc
            auc.loc[:, 'subject']=subject
            auc.loc[:, 'cluster']=cluster
            auc.set_index(['subject', 'cluster'], append=True, inplace=True)
            auc = auc.reorder_levels(['subject', 'cluster', 'confidence', 'response'])
            AUCS.append(auc)
    return AUCS


def sanity_check_auc(meta, X):
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    vals = np.ones((4,4))*np.nan
    combinations = [(2, -1), (1, -1), (1, 1), (2, 1)]
    for i, c1 in enumerate(combinations):
        for j, c2 in enumerate(combinations):
            c1idx = (meta.confidence==c1[0]) & (meta.response==c1[1])
            c2idx = (meta.confidence==c2[0]) & (meta.response==c2[1])
            values = np.concatenate([X[c1idx], X[c2idx]])
            #print(values.shape, sum(c1idx), sum(c2idx))
            labels = values*0
            labels[:sum(c1idx)] = 1
            vals[i,j] = roc_auc_score(labels, values)
    index = pd.MultiIndex.from_tuples(combinations, names=['confidence', 'response'])
    vals = pd.DataFrame(vals, index=index, columns=index)    
    return vals


def ensure_iter(input):
    if isinstance(input, str):
        yield input
    else:
        try:
            for item in input:
                yield item
        except TypeError:
            yield input


def combine_signals(signal, epoch):
    print(',a;lsx,l;as,lx;,')
    import glob
    globstr = '/mnt/homes/home028/nwilming/conf_meg/sr_decoding/concat_S*-%s_*var*%s-decoding.hdf'%(signal, epoch)
    print(globstr)
    files = glob.glob(globstr)
    frames = []
    for f in files:
        df = pd.read_hdf(f)
        cluster = [x.replace('[','').replace(']', '').replace("'","") for x in df.cluster.values]
        df.loc[:, cluster] = cluster
        df.loc[:, 'signal'] = signal
        df.loc[:, 'epoch'] = epoch
        df.loc[:, 'subject'] = df.sub
        frames.append(df)
    return pd.concat(frames)
