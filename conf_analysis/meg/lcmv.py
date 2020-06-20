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
import datetime


memory = Memory(cachedir=metadata.cachedir)
path = "/home/nwilming/conf_meg/sr_labeled/"
subjects_dir = "/home/nwilming/fs_subject_dir"
trans_dir = "/home/nwilming/conf_meg/trans"


def set_n_threads(n):
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OMP_NUM_THREADS"] = str(n)


def modification_date(filename):
    try:
        t = os.path.getmtime(filename)
        return datetime.datetime.fromtimestamp(t)
    except OSError:
        return datetime.datetime.strptime("19700101", "%Y%m%d")


def submit(older_than="201911010000", only_glasser=False):
    from pymeg import parallel
    from itertools import product

    cnt = 1
    older_than = datetime.datetime.strptime(older_than, "%Y%m%d%H%M")
    cnt = 1
    #for subject, session, epoch, signal in [
    #   [3, 0, "stimulus", "F"],       
    #   [9, 2, "stimulus", "F"],
    #   [10, 0, "response", "F"],       
    #]:
    for subject, session, epoch, signal in product(
        range(1, 16), range(4), ["response"], ["F", "LF"]
    ):
        mod_time = [
            modification_date(x)
            for x in lcmvfilename(
                subject, session, signal, epoch, chunk="all", only_glasser=only_glasser
            )
        ]
        # if(any([x > older_than for x in mod_time])):
        #    print("Skipping %i %i %s %s because existing output is newer than requested date" % (
        #        subject, session, epoch, signal))
        #    continue
        # def extract(
        #    subject,
        #    session,
        #    epoch_type="stimulus",
        #    signal_type="BB",
        #    only_glasser=False,
        print("Submitting %i %i %s %s" % (subject, session, epoch, signal))
        parallel.pmap(
            extract,
            [(subject, session, epoch, signal, only_glasser)],
            walltime="10:00:00",
            memory=40,
            nodes=1,
            tasks=5,
            env="py36",
            name="SR" + str(subject) + "_" + str(session) + epoch,
            ssh_to=None,
        )
        # if np.mod(cnt, 15) == 0:
        #    import time
        #    time.sleep(60 * 30)
        cnt += 1


def lcmvfilename(subject, session, signal, epoch_type, chunk=None, only_glasser=False):
    try:
        makedirs(path)
    except:
        pass
    if chunk is None:
        filename = "S%i-SESS%i-%s-%s-lcmv.hdf" % (subject, session, epoch_type, signal)
    elif chunk is "all":
        import glob

        filename = "S%i-SESS%i-%s-%s-chunk*-lcmv.hdf" % (
            subject,
            session,
            epoch_type,
            signal,
        )
        filenames = glob.glob(join(path, filename))
        return filenames
    else:
        filename = "S%i-SESS%i-%s-%s-chunk%i-lcmv.hdf" % (
            subject,
            session,
            epoch_type,
            signal,
            chunk,
        )
    if only_glasser:
        filename = "ogl_" + filename
        return join(path, "ogl", filename)
    else:
        return join(path, filename)


def get_stim_epoch(subject, session):
    epochs, meta = preprocessing.get_epochs_for_subject(
        subject, "stimulus", sessions=session
    )
    epochs = epochs.pick_channels([x for x in epochs.ch_names if x.startswith("M")])
    id_time = (-0.25 <= epochs.times) & (epochs.times <= 0)
    means = epochs._data[:, :, id_time].mean(-1)
    epochs._data -= means[:, :, np.newaxis]
    data_cov = pymeglcmv.get_cov(epochs, tmin=0, tmax=1.35)
    return data_cov, epochs


def get_response_epoch(subject, session):
    epochs, meta = preprocessing.get_epochs_for_subject(
        subject, "stimulus", sessions=session
    )
    epochs = epochs.pick_channels([x for x in epochs.ch_names if x.startswith("M")])
    response, meta = preprocessing.get_epochs_for_subject(
        subject, "response", sessions=session
    )
    response = response.pick_channels(
        [x for x in response.ch_names if x.startswith("M")]
    )
    # Find trials that are present in both time periods
    overlap = list(set(epochs.events[:, 2]).intersection(set(response.events[:, 2])))
    epochs = epochs[[str(l) for l in overlap]]
    response = response[[str(l) for l in overlap]]
    id_time = (-0.25 <= epochs.times) & (epochs.times <= 0)
    means = epochs._data[:, :, id_time].mean(-1)
    epochs._data -= means[:, :, np.newaxis]

    response._data = response._data - means[:, :, np.newaxis]

    # Now also baseline stimulus period
    epochs, meta = preprocessing.get_epochs_for_subject(
        subject, "stimulus", sessions=session
    )
    epochs = epochs.pick_channels([x for x in epochs.ch_names if x.startswith("M")])
    id_time = (-0.25 <= epochs.times) & (epochs.times <= 0)
    means = epochs._data[:, :, id_time].mean(-1)
    epochs._data -= means[:, :, np.newaxis]
    data_cov = pymeglcmv.get_cov(epochs, tmin=0, tmax=1.35)
    return data_cov, response


def get_trans(subject, session):
    """
    Return filename of transformation for a subject
    """
    file_ident = "S%i-SESS%i" % (subject, session)
    return join(trans_dir, file_ident + "-trans.fif")


def get_leadfield(subject, session, head_model="three_layer"):
    """
    Compute leadfield with presets for this subject

    Parameters
    head_model : str, 'three_layer' or 'single_layer'
    """
    raw_filename = metadata.get_raw_filename(subject, session)
    epoch_filename = metadata.get_epoch_filename(subject, session, 0, "stimulus", "fif")
    trans = get_trans(subject, session)

    return pymegsr.get_leadfield(
        "S%02i" % subject,
        raw_filename,
        epoch_filename,
        trans,
        conductivity=(0.3, 0.006, 0.3),
        bem_sub_path="bem_ft",
        njobs=4,
    )


def extract(
    subject,
    session,
    epoch_type="stimulus",
    signal_type="BB",
    only_glasser=False,
    BEM="three_layer",
    debug=False,
    chunks=100,
    njobs=4,
):
    mne.set_log_level("WARNING")
    pymeglcmv.logging.getLogger().setLevel(logging.INFO)
    set_n_threads(1)

    logging.info("Reading stimulus data")
    if epoch_type == "stimulus":
        data_cov, epochs = get_stim_epoch(subject, session)
    elif epoch_type == "response":
        data_cov, epochs = get_response_epoch(subject, session)
    else:
        raise RuntimeError("Did not recognize epoch")

    logging.info("Setting up source space and forward model")

    forward, bem, source = get_leadfield(subject, session, BEM)

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
    # Now chunk Reconstruction into blocks of ~100 trials to save Memory
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

    events = epochs.events[:, 2]
    data = []
    filters = pymeglcmv.setup_filters(epochs.info, forward, data_cov, None, labels)
    
    set_n_threads(1)

    for i in range(0, len(events), chunks):
        filename = lcmvfilename(subject, session, signal_type, epoch_type, chunk=i, 
            only_glasser=only_glasser)
        logging.info(filename)
        # if os.path.isfile(filename):
        #    continue
        if signal_type == "BB":
            logging.info("Starting reconstruction of BB signal")
            M = pymeglcmv.reconstruct_broadband(
                filters,
                epochs.info,
                epochs._data[i : i + chunks],
                events[i : i + chunks],
                epochs.times,
                njobs=1,
            )
        else:
            logging.info("Starting reconstruction of TFR signal")
            M = pymeglcmv.reconstruct_tfr(
                filters,
                epochs.info,
                epochs._data[i : i + chunks],
                events[i : i + chunks],
                epochs.times,
                est_args=tfr_params[signal_type],
                njobs=4,
            )
        M.to_hdf(filename, "epochs", mode="w")
    set_n_threads(njobs)


def get_filter(
    subject,
    session,
    epoch_type="stimulus",    
    only_glasser=False,
    BEM="three_layer",    
):
    mne.set_log_level("WARNING")
    pymeglcmv.logging.getLogger().setLevel(logging.INFO)
    set_n_threads(1)

    logging.info("Reading stimulus data")
    if epoch_type == "stimulus":
        data_cov, epochs = get_stim_epoch(subject, session)
    elif epoch_type == "response":
        data_cov, epochs = get_response_epoch(subject, session)
    else:
        raise RuntimeError("Did not recognize epoch")

    logging.info("Setting up source space and forward model")

    forward, bem, source = get_leadfield(subject, session, BEM)

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
    # Now chunk Reconstruction into blocks of ~100 trials to save Memory
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

    events = epochs.events[:, 2]
    data = []
    filters = pymeglcmv.setup_filters(epochs.info, forward, data_cov, None, labels)
    return filters
