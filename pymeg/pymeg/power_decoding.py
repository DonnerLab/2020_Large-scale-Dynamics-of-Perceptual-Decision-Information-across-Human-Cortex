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
import numpy as np
import pandas as pd

from functools import partial
from itertools import product

from sklearn import linear_model, svm
from sklearn.model_selection import cross_validate, cross_val_predict

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import RandomOverSampler


def decoders():
    return {
        "SSD": continuous_decoder,
        "SSD_acc_contrast": partial(continuous_decoder, target="acc_contrast"),
        "MIDC_split": partial(category_decoder, splitmc=True, target="response"),
    }


def apply_across_areas(
    decoder,
    meta,
    filenames,
    clusters,
    decoder,
    hemis=["Pair", "Lateralized", "Averaged"],
    ntasks=-1,
):
    """
    Run a decoder across a set of clusters, time points and hemis.

    Args:
        meta: pd.DataFrame
            Contains the target colum, indexed by trial.
        filenames: list
            Filenames containing source reconstructed aggregates.
        decoder: function
            Function that takes meta data, aggregate data, areas
            and latency as input and returns decoding scores
            as data frame.
        latencies: None or list
            List of latencies to decode across. If None
            all time points in agg will be used.

    """
    set_n_threads(1)
    from multiprocessing import Pool
    from pymeg import aggregate_sr as asr

    areas = clusters.keys()
    args = []
    for area, hemi in product(areas, hemis):
        if hemi == "Pair":
            area = [area + "_RH", area + "_LH"]
        a = (meta, asr.delayed_agg(filenames, hemi=hemi, cluster=area), decoder, hemi)
        args.append(a)
    print("Processing %i tasks" % (len(args)))
    from contextlib import closing

    scores = []

    with closing(Pool(ntasks, maxtasksperchild=1)) as p:
        sc = p.starmap(_par_apply_decode, args)
        sc = [s for s in sc if s is not None]
    scores.extend(sc)
    scores = pd.concat(scores)

    p.terminate()
    return scores


def load_and_apply(meta, delayed_agg, decoder, hemi=None):
    agg = delayed_agg()
    scores = apply_across_time(
        meta, agg, decoders()[decoder], latencies=latencies, hemi=hemi
    )
    return scores


def apply_across_time(meta, agg, decoder, latencies=None, hemi=None):
    """Run a decoder across a set of latencies.

    Args:
        meta: pd.DataFrame
            Contains the target colum, indexed by trial.
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

    start = time.time()
    if latencies is None:
        latencies = agg.columns.get_level_values("time").values

    area = np.unique(agg.index.get_level_values("cluster"))
    scores = []

    for latency in latencies:
        s = decoder(meta, agg[latency], area)
        s.loc[:, "cluster"] = str(area)
        s.loc[:, "latency"] = latency
        scores.append(s)
    res = pd.concat(scores)
    res.loc[:, "hemi"] = hemi
    logging.info(
        "Applying decoder %s across N=%i latencies took %3.2fs"
        % (decoder, len(latencies), time.time() - start)
    )
    return res


def continuous_decoder(meta, data, area, target):
    """
    Decode continuous signals via Regression.

    Each frequency and brain area is one feature, contrast is the target.
    Args:
        meta: DataFrame
            Metadata frame that contains meta data per row
        data: Aggregate data frame
            This frame is transformed into a trial x (freq feature matrix*area).
        area: List or str
            Which areas to use for decoding. Multiple areas provide
            independent features per observation        
        target_value: str
            Which target to decode (refers to a col. in meta)

    """

    from sklearn.metrics import make_scorer
    from scipy.stats import linregress

    slope_scorer = make_scorer(lambda x, y: linregress(x, y)[0])
    corr_scorer = make_scorer(lambda x, y: np.corrcoef(x, y)[0, 1])
    # Turn data into (trial X Frequency)
    data = []
    for a in ensure_iter(area):
        x = pd.pivot_table(
            data.query('cluster=="%s"' % a), index="trial", columns="freq"
        )
        data.append(x)
    data = pd.concat(data, 1)
    target = meta.loc[:, target]
    target = target.loc[data.index]

    metrics = {
        "explained_variance": "explained_variance",
        "r2": "r2",
        "slope": slope_scorer,
        "corr": corr_scorer,
    }
    classifiers = {"Ridge": linear_model.RidgeCV()}
    scores = []
    for name, clf in list(classifiers.items()):
        clf = Pipeline([("Scaling", StandardScaler()), (name, clf)])
        score = cross_validate(
            clf,
            data.values,
            target,
            cv=10,
            scoring=metrics,
            return_train_score=False,
            n_jobs=1,
        )
        del score["fit_time"]
        del score["score_time"]
        score = {k: np.mean(v) for k, v in list(score.items())}
        score["Classifier"] = name
        scores.append(score)
    return pd.DataFrame(scores)


def category_decoder(
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
            split_label = "R1"
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


def set_n_threads(n):
    import os

    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OMP_NUM_THREADS"] = str(n)


def ensure_iter(input):
    if isinstance(input, str):
        yield input
    else:
        try:
            for item in input:
                yield item
        except TypeError:
            yield input
