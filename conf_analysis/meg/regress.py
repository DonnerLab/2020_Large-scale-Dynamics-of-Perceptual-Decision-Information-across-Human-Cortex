#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import numpy as np
import pandas as pd

from os.path import join
from glob import glob

from pymeg import aggregate_sr as asr
from conf_analysis.behavior import metadata
from conf_analysis.meg import preprocessing
from conf_analysis.meg import decoding_analysis as da

from sklearn import linear_model, discriminant_analysis, svm
from sklearn.model_selection import (
    cross_validate,
    cross_val_predict,
    RandomizedSearchCV,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import randint as sp_randint
import pickle


from joblib import Memory

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

n_jobs = 1


def shuffle(data):
    return data[np.random.permutation(np.arange(len(data)))]


def eval_areas(
    cache_key, motor_latency=1.2, latency_stim=np.arange(-0.1, 0.35, 1 / 60.0)
):
    scores = []
    for area in ["JWG_M1", "JWG_IPS_PCeS", "JWG_aIPS", "vfcFEF"]:
        sc, _, _, _ = eval_stim_latency(
            area,
            motor_latency=motor_latency,
            latency_stim=latency_stim,
            motor_area=area,
        )
        scores.append(sc)
    scores = pd.concat(scores, 0)
    pickle.dump(
        {"scores": scores},
        open(
            "/home/nwilming/conf_analysis/results/%s_Xarea_stim_latency.pickle"
            % cache_key,
            "wb",
        ),
    )


def eval_all(cache_key):
    """
    Perform coupling analysis for samples up to a specific sample
    with specific stim latency.

    Evaluate copuling for all motor latencies and a range of stim latencies

    """
    # fmt: off
    motor_latencies = np.array(
      [0.   , 0.017, 0.033, 0.05 , 0.067, 0.083, 0.1  , 0.117, 0.133,
       0.15 , 0.167, 0.183, 0.2  , 0.217, 0.233, 0.25 , 0.267, 0.283,
       0.3  , 0.317, 0.333, 0.35 , 0.367, 0.383, 0.4  , 0.417, 0.433,
       0.45 , 0.467, 0.483, 0.5  , 0.517, 0.533, 0.55 , 0.567, 0.583,
       0.6  , 0.617, 0.633, 0.65 , 0.667, 0.683, 0.7  , 0.717, 0.733,
       0.75 , 0.767, 0.783, 0.8  , 0.817, 0.833, 0.85 , 0.867, 0.883,
       0.9  , 0.917, 0.933, 0.95 , 0.967, 0.983, 1.   , 1.017, 1.033,
       1.05 , 1.067, 1.083, 1.1  ])
    # fmt: on
    latency_stim = np.arange(-0.1, 0.35, 1 / 60.0)
    scores = []
    try:
        for motor_latency in motor_latencies:
            for area in ["JWG_M1", "JWG_IPS_PCeS", "JWG_aIPS", "vfcFEF"]:
                sc, _, _, _ = eval_stim_latency(
                    area,
                    motor_latency=motor_latency,
                    latency_stim=latency_stim,
                    motor_area=area,
                )
                scores.append(sc)
    finally:
        scores = pd.concat(scores, 0)
        pickle.dump(
            {"scores": scores},
            open(
                "/home/nwilming/conf_analysis/results/%s_Xarea_stim_latency.pickle"
                % cache_key,
                "wb",
            ),
        )
    return scores


def preheat_get_motor_prediction(subject, latency_motor, cluster):
    return get_motor_prediction(subject, latency_motor, cluster=cluster)

def eval_stim_latency(
    cache_key,
    motor_latency=1.4,
    baseline=False,
    latency_stim=np.arange(-0.1, 0.4, 1 / 60.0),
    n_jobs=12,
    motor_area="JWG_M1",
):
    from joblib import Parallel, delayed
    import pickle
    #motor, evals = get_motor_prediction(subject, latency_motor, cluster=motor_area)

    args = list(
        delayed(preheat_get_motor_prediction)(i, motor_latency, motor_area)
        for i in range(1, 16)
    )
    print(args)
    results = Parallel(n_jobs=n_jobs)(args)


    args = list(
        delayed(eval_all_subs)(motor_latency, i, baseline, motor_area)
        for i in latency_stim
    )
    results = Parallel(n_jobs=n_jobs)(args)
    sc = []
    weights = {}
    shuff_weights = {}
    iweights = {}
    intweights = {}
    for j, i in enumerate(latency_stim):
        results[j][0].loc[:, "latency"] = i
        sc.append(results[j][0])
        weights[i] = results[j][1]
        shuff_weights[i] = results[j][2]
        iweights[i] = results[j][3]

    scores = pd.concat(sc, 0)
    pickle.dump(
        {
            "scores": scores,
            "weights": weights,
            "shuffled_weights": shuff_weights,
            "1smp_weights": iweights,
        },
        open(
            "/home/nwilming/conf_analysis/results/%s_stim_latency.pickle" % cache_key,
            "wb",
        ),
    )
    return scores, weights, shuff_weights, iweights


def get_cache_key(cache_key):
    o = pickle.load(
        open(
            "/home/nwilming/conf_analysis/results/%s_stim_latency.pickle" % cache_key,
            "rb",
        )
    )
    return o["scores"], o["weights"], o["shuffled_weights"], o["1smp_weights"]


def eval_all_subs(
    latency_motor=1.4, latency_stim=0.18, baseline_correct=False, motor_area="JWG_M1"
):
    scores = []
    shuffled_scores = []
    iscores = []
    weights = []
    sweights = []
    iweights = []
    integrator_weights = []
    integrator_scores = []
    for subject in range(1, 16):
        s, w, ss, sw, si, iw, sint, wint = eval_coupling(
            subject,
            latency_motor=latency_motor,
            latency_stim=latency_stim,
            baseline_correct=baseline_correct,
            motor_area=motor_area,
        )
        scores.append(s)
        weights.append(w)
        shuffled_scores.append(ss)
        sweights.append(sw)
        iscores.append(si)
        iweights.append(iw)
        integrator_scores.append(sint)
        integrator_weights.append(wint)
    scores = pd.DataFrame(
        {
            "corr": scores,
            "1smp_corr": iscores,
            "integrator_corr": integrator_scores,
            "shuff_corr": shuffled_scores,
        }
    )
    scores.loc[:, "subject"] = np.arange(1, 16)
    scores.loc[:, "motor_area"] = motor_area
    return scores, weights, sweights, iweights


def weight_to_act(X, w, i):
    w = np.concatenate((w, [i]))
    SXn = np.cov(X.T)
    # w = w[:, np.newaxis]
    return np.dot(SXn, w)


def eval_coupling(
    subject,
    latency_motor=1.4,
    latency_stim=0.18,
    baseline_correct=False,
    motor_area="JWG_M1",
):
    motor, evals = get_motor_prediction(subject, latency_motor, cluster=motor_area)
    print("S %i AUC:" % subject, evals["test_roc_auc"])
    lodds = np.log(motor.loc[:, 0] / motor.loc[:, 1]).values
    X, tpc, freqs = build_design_matrix(
        subject,
        motor.index,
        "vfcPrimary",
        latency_stim,
        add_contrast=False,
        zscore=True,
        freq_bands=[45, 65],
        # freq_bands=[0, 8, 39, 61, 100],
    )
    if baseline_correct:
        base, tpc, freqs = build_design_matrix(
            subject,
            motor.index,
            "vfcPrimary",
            0,
            add_contrast=False,
            zscore=True,
            freq_bands=[45, 65],
            # freq_bands=[0, 8, 39, 61, 100],
        )
        X = X - base
    tpc = np.array(tpc)
    freqs = np.array(freqs)
    # Make sure there are no samples in the future
    idcol = np.array(list(tpc < latency_motor) + [True])
    print("Kicking out %s columns" % sum(~idcol))
    X = X[:, idcol]
    tpc = tpc[idcol[:-1]]
    freqs = freqs[idcol[:-1]]
    score, weights, intercept = coupling(lodds, X, n_iter=250, pcdist=sp_randint(5, 40))
    shuffled_score, shuffled_weights, s_intercept = coupling(
        shuffle(lodds), X, n_iter=250, pcdist=sp_randint(5, 40)
    )
    inst_corr, iweights, _ = coupling(lodds, X[:, -2:], n_iter=250, pcdist=None)
    integrator_corr, integrator_weights, _ = coupling(
        lodds, X.sum(1).reshape(-1, 1), n_iter=250, pcdist=None
    )

    return (
        score,
        weight_to_act(X, weights, intercept),
        shuffled_score,
        weight_to_act(X, shuffled_weights, s_intercept),
        inst_corr,
        iweights,
        integrator_corr,
        integrator_weights,
    )


@memory.cache()
def get_motor_prediction(subject, latency, cluster="JWG_M1"):
    # First load low level averaged stimulus data
    filenames = glob(join(inpath, "S%i_*_%s_agg.hdf" % (subject, "stimulus")))
    data = asr.delayed_agg(filenames, hemi="Lateralized", cluster=cluster)()
    meta = da.augment_meta(da.preprocessing.get_meta_for_subject(subject, "stimulus"))
    scores = da.midc_decoder(
        meta,
        data,
        cluster,
        latency=latency,
        splitmc=False,
        target_col="response",
        predict=True,
    )
    eval_scores = da.midc_decoder(
        meta,
        data,
        cluster,
        latency=latency,
        splitmc=False,
        target_col="response",
        predict=False,
    )
    return scores, eval_scores



def coupling_all_subjects_all_t():
    couplings = []
    for subject in np.arange(1, 16):
        motor_predictions = get_all_motor_prediction(subject, cluster='JWG_M1')
        coupling = get_all_coupling(subject, motor_predictions)
        couplings.append(coupling)
        cps = pd.concat(couplings)
        cps.to_hdf('/home/nwilming/all_couplings_M1_new.hdf', 'df')



@memory.cache()
def get_all_motor_prediction(
    subject, 
    latencies=np.arange(0, 1.2, 1 / 60), 
    cluster="JWG_M1",
    n_jobs=10
):
    from joblib import Parallel, delayed
    # First load low level averaged stimulus data
    filenames = glob(join(inpath, "S%i_*_%s_agg.hdf" % (subject, "stimulus")))
    data = asr.delayed_agg(filenames, hemi="Lateralized", cluster=cluster)()
    meta = da.augment_meta(da.preprocessing.get_meta_for_subject(subject, "stimulus"))
    eval_scores = {}, 

    args = list(
        delayed(da.midc_decoder)(meta, data, cluster, latency, False, False, 'response', True)
        for latency in latencies   
    )
    results = Parallel(n_jobs=n_jobs)(args)
    results = pd.concat(results)
    results.loc[:, 'subject'] = subject
    return results


def get_all_coupling(subject, motor_predictions, cluster='vfcPrimary', n_jobs=10):
    from joblib import Parallel, delayed
    filenames = glob(join(inpath, "S%i_*_%s_agg.hdf" % (subject, "stimulus")))
    data = asr.delayed_agg(filenames, hemi="Averaged", cluster=cluster)()
    data = data.query('45<=freq & freq<65')
    X = pd.pivot_table(data.loc[:, 0:1.4], index='trial')
    X = (X - X.mean(0)) / X.std(0)
    tasks = []
    temp_idx = np.arange(0, 1, .10)
    cache = {}
    for motor_latency, motor_data in motor_predictions.groupby('latency'):
        target =  np.log(motor_data.loc[:, 0] / motor_data.loc[:, 1])
        for readout_latency in np.arange(-0.2, 0.21, 1/60):
            temp_selector = temp_idx + readout_latency
            temp_selector = temp_selector[temp_selector<motor_latency]
            if len(temp_selector) == 0:
                continue        
            times = X.columns.get_level_values('time').values   
            temp_selector = [times[np.argmin(np.abs(times-t))] for t in temp_selector]
            Xb = X.loc[:, temp_selector]            
            tasks.append(delayed(get_one_coupling)(subject, motor_latency, readout_latency, Xb, target))
            
    print('Prepared %i tasks'%len(tasks))
    results = Parallel(n_jobs=n_jobs)(tasks)
    return pd.DataFrame(results)


def get_one_coupling(subject, motor_latency, readout_latency, Xb, target):
    Xb_offset = np.concatenate([Xb.values, Xb.values[:,0]*0+1])
    Xb_integrate_offset = np.concatenate([Xb.values.sum(1), Xb.values[:,0]*0+1])
    Xb_last_samp = np.concatenate([Xb.values[:, -1], Xb.values[:,0]*0+1])
    score, _, _ = coupling(target, Xb_offset, n_iter=150, pcdist=sp_randint(5, 40))        
    integrator, _, _ = coupling(target, Xb_integrate_offset, n_iter=150, pcdist=sp_randint(5, 40))
    last_samp, _, _ = coupling(target, Xb_last_samp, n_iter=150, pcdist=sp_randint(5, 40))                        
    return {
        'subject':subject,
        'motor_latency':motor_latency,
        'readout_latency':readout_latency,
        'weighted_score':score,
        'integrator_score': integrator,
        'last_sample': last_samp,
       }

@memory.cache()
def build_design_matrix(
    subject,
    trial_index,
    cluster,
    latency,
    add_contrast=False,
    zscore=True,
    freq_bands=None,
):
    filenames = glob(join(inpath, "S%i_*_%s_agg.hdf" % (subject, "stimulus")))
    data = asr.delayed_agg(filenames, hemi="Averaged", cluster=cluster)()
    X, time_per_col = prep_low_level_data(
        cluster, data, 0, latency, trial_index, freq_bands=freq_bands
    )
    cols = X.columns.values
    index = X.index.values
    X = X.values

    if add_contrast:
        meta = da.preprocessing.get_meta_for_subject(subject, "stimulus")
        meta.set_index("hash", inplace=True)
        meta = meta.loc[trial_index]
        cvals = np.stack(meta.contrast_probe)
        X = np.hstack([X, cvals])
    if zscore:
        X = (X - X.mean(0)) / X.std(0)
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    return X, time_per_col, cols


def prep_low_level_data(areas, data, peak, latency, trial_index, freq_bands=None):
    lld = []
    times = data.columns.get_level_values("time").values
    time_per_col = []
    for s in np.arange(0, 1, 0.1) + latency:
        # Turn data into (trial X Frequency)
        target_time_point = times[np.argmin(abs(times - s))]
        for a in da.ensure_iter(areas):
            x = pd.pivot_table(
                data.query('cluster=="%s"' % a),
                index="trial",
                columns="freq",
                values=target_time_point,
            )
            if freq_bands is not None:
                x = (
                    x.T.groupby(
                        pd.cut(
                            x.T.index,
                            freq_bands,
                            labels=np.array(freq_bands)[:-1] + np.diff(freq_bands) / 2,
                        )
                    )
                    .mean()
                    .T
                )
            lld.append(x.loc[trial_index])
            time_per_col.extend([s] * x.shape[1])
    return pd.concat(lld, 1), time_per_col



@memory.cache()
def trial_time_design_matrix(
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
    cols = X.columns.values
    index = X.index.values
    X = X.values
    meta = da.preprocessing.get_meta_for_subject(subject, "stimulus")
    meta.set_index("hash", inplace=True)
    meta = meta.loc[trial_index]
    if zscore:
        X = (X - X.mean(0)) / X.std(0)

    return X, time_per_col, cols, meta


def trial_time_data_prep(area, data, peak, trial_index, freq_band=(0, 150)):
    lld = []
    times = data.columns.get_level_values("time").values
    data = data.query('cluster=="%s"' % a).groupby(['trial']).mean()
    data = pd.pivot_table(
                data,
                index="trial",                                
            )
    return data, data.columns.values


def coupling(target, X, n_iter=50, pcdist=sp_randint(5, 40)):
    """
    
    """
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
    from sklearn.metrics import roc_auc_score, mean_squared_error
    from sklearn.decomposition import PCA
    from sklearn.utils import shuffle
    from imblearn.pipeline import Pipeline
    from sklearn.metrics.scorer import make_scorer

    corr_scorer = make_scorer(
        lambda x, y: np.corrcoef(x, y)[0, 1], greater_is_better=True
    )

    classifier = Pipeline(
        [
            # ("Scaling", StandardScaler()),
            # ("PCA", PCA(n_components=0.3, svd_solver="full")),
            ("linear_regression", Ridge(fit_intercept=False))
        ]
    )
    if pcdist is not None:
        classifier = RandomizedSearchCV(
            classifier,
            param_distributions={
                # "PCA__n_components": pcdist,
                "linear_regression__alpha": sp_randint(1, 100000)
            },
            n_iter=n_iter,
            cv=3,
        )
    else:
        classifier = RandomizedSearchCV(
            classifier,
            param_distributions={"linear_regression__alpha": sp_randint(1, 100000)},
            n_iter=n_iter,
            cv=3,
        )
    scores = cross_validate(
        classifier,
        X,
        target,
        cv=3,
        scoring=corr_scorer,
        return_train_score=False,
        return_estimator=True,
        n_jobs=1,
    )

    coefs = np.stack(
        [o.best_estimator_.steps[-1][1].coef_ for o in scores["estimator"]]
    )
    coefs = coefs.mean(0)
    return scores["test_score"].mean(), coefs[:-1], coefs[-1]
