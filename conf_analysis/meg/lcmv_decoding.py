import logging
import mne
import numpy as np
import os

from conf_analysis.behavior import metadata
from conf_analysis.meg import preprocessing

from joblib import Memory

from os import makedirs
from os.path import join
from pymeg import lcmv as pymeglcmv
from pymeg import source_reconstruction as pymegsr
from pymeg import decoding
import datetime

import pandas as pd

from conf_analysis.meg.lcmv import (
    get_stim_epoch,
    get_response_epoch,
    get_trans,
    get_leadfield,
)
from conf_analysis.meg.decoding_analysis import augment_meta

memory = Memory(cachedir=metadata.cachedir)
path = "/home/nwilming/conf_meg/sr_labeled/"
subjects_dir = "/home/nwilming/fs_subject_dir"
trans_dir = "/home/nwilming/conf_meg/trans"


def set_n_threads(n):
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OMP_NUM_THREADS"] = str(n)


areas_to_labels = {
    "JWG_M1": ["lr_M1"],
    
    "pgACC_9m": ["9m_ROI"],
    "pgACC_p24": ["p24_ROI"],
    "pgACC_d32": ["d32_ROI"],
    "pgACC_a24": ["a24_ROI"],

    "dlpfc_8C": ["8C_ROI"],
    "dlpfc_8Av": ["8Av_ROI"],
    "dlpfc_i6-8": ["i6-8_ROI"],
    "dlpfc_s6-8": ["s6-8_ROI"],
    "dlpfc_SFL": ["SFL_ROI"],
    "dlpfc_8BL": ["8BL_ROI"],
    "dlpfc_9p": ["9p_ROI"],
    "dlpfc_9a": ["9a_ROI"],
    "dlpfc_8Ad": ["8Ad_ROI"],
    "dlpfc_p9-46v": ["p9-46v_ROI"],
    "dlpfc_46": ["46_ROI"],
    "dlpfc_9-46d": ["9-46d_ROI"],
    "dlpfc_a9-46d": ["a9-46v_ROI"],

    "vfcIPS01": [
        u"lh.wang2015atlas.IPS0-lh",
        u"rh.wang2015atlas.IPS0-rh",
        u"lh.wang2015atlas.IPS1-lh",
        u"rh.wang2015atlas.IPS1-rh",
    ],
    "vfcIPS23": [
        u"lh.wang2015atlas.IPS2-lh",
        u"rh.wang2015atlas.IPS2-rh",
        u"lh.wang2015atlas.IPS3-lh",
        u"rh.wang2015atlas.IPS3-rh",
    ],

}


def submit(only_glasser=False):
    from pymeg import parallel
    from itertools import product

    for area in list(areas_to_labels.keys()):
        for subject in range(1, 16):
            print("Submitting S%i -> %s" % (subject, area))
            parallel.pmap(
                decode,
                [(subject, area)],
                walltime="12:00:00",
                memory=50,
                nodes=1,
                tasks=6,
                env="py36",
                name="iDCD" + str(subject) + area,
                ssh_to=None,
            )
            
                


def get_labels(subject, only_glasser):
    if not only_glasser:
        labels = pymegsr.get_labels(
            subject="S%02i" % subject,
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
    else:
        labels = pymegsr.get_labels(
            subject="S%02i" % subject,
            filters=["select_nothing"],
            annotations=["HCPMMP1"],
        )
    return labels


@memory.cache
def decode(
    subject,
    area,
    epoch_type="stimulus",
    only_glasser=False,
    BEM="three_layer",
    debug=False,
    target="response",
):
    mne.set_log_level("WARNING")
    pymeglcmv.logging.getLogger().setLevel(logging.INFO)
    set_n_threads(1)
    labels = get_labels(subject, only_glasser)
    labels = [x for x in labels if any([cl for cl in areas_to_labels[area] if cl in x.name])]
    print(labels)
    if len(labels) < 1:
        raise RuntimeError('Expecting at least two labels')
    label = labels.pop()
    for l in labels:
        label += l
    #label = labels[0] + labels[1]

    print('Selecting this label for area %s:'%area, label)

    #return

    logging.info("Reading stimulus data")
    if epoch_type == "stimulus":
        # data_cov, epochs = get_stim_epoch(subject, session)
        data = [get_stim_epoch(subject, i) for i in range(4)]
    elif epoch_type == "response":
        # data_cov, epochs = get_response_epoch(subject, session)
        data = [get_response_epoch(subject, i) for i in range(4)]
    else:
        raise RuntimeError("Did not recognize epoch")


    logging.info("Setting up source space and forward model")

    fwds = [get_leadfield(subject, session, BEM)[0] for session in range(4)]

    fois = np.arange(10, 150, 5)
    lfois = np.arange(1, 10, 1)
    tfr_params = {
        "F": {
            "foi": fois,
            "cycles": fois * 0.1,
            "time_bandwidth": 2,
            "n_jobs": 1,
            "est_val": fois,
            "est_key": "F",
        },
        "LF": {
            "foi": lfois,
            "cycles": lfois * 0.25,
            "time_bandwidth": 2,
            "n_jobs": 1,
            "est_val": lfois,
            "est_key": "LF",
        },
    }

    events = [d[1].events[:, 2] for d in data]
    filters = []
    for (data_cov, epochs), forward in zip(data, fwds):
        filters.append(
            pymeglcmv.setup_filters(epochs.info, forward, data_cov, None, [label])
        )
    set_n_threads(1)

    F_tfrdata, events, F_freq, times = decoding.get_lcmv(
        tfr_params["F"], [d[1] for d in data], filters, njobs=6
    )
    LF_tfrdata, events, LF_freq, times = decoding.get_lcmv(
        tfr_params["LF"], [d[1] for d in data], filters, njobs=6
    )

    tfrdata = np.hstack([F_tfrdata, LF_tfrdata])
    del LF_tfrdata, F_tfrdata
    freq = np.concatenate([F_freq, LF_freq])
    meta = augment_meta(
        preprocessing.get_meta_for_subject(
            subject, epoch_type, sessions=range(4)
        ).set_index("hash")
    )

    # Kick out trials with RT < 0.225    
    choice_rt = meta.choice_rt
    valid_trials = choice_rt[choice_rt>=0.225].index.values        
    valid_trials = np.isin(events, valid_trials)
    tfrdata = tfrdata[valid_trials]
    events = events[valid_trials]
    # How many kicked out?
    n_out = (~valid_trials).sum()
    n_all = len(events)    
    print('Kicking out %i/%i (%0.2f percent) trials due to RT'%(n_out, n_all, n_out/n_all))

    all_s = []
    for target in ["response", "unsigned_confidence", "signed_confidence"]:
        fname = "/home/nwilming/S%i_trg%s_ROI_%s.hdf" % (subject, target, area)
        try:
            k = pd.read_hdf(fname)
        except FileNotFoundError:
            target_vals = meta.loc[:, target]
            dcd = decoding.Decoder(target_vals)
            k = dcd.classify(
                tfrdata, times, freq, events, area, 
                average_vertices=False, use_phase=True
            )
            k.loc[:, "target"] = target
            k.to_hdf(fname, "df")
        all_s.append(k)
    all_s = pd.concat(all_s)
    all_s.loc[:, 'ROI'] = area
    all_s.to_hdf("/home/nwilming/S%i_ROI_%s.hdf" % (subject, area), "df")
    return k
