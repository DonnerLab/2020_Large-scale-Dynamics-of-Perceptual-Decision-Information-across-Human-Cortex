'''
Do some decoding stuff.
'''
from conf_analysis.meg import preprocessing, tfr_analysis
from conf_analysis.behavior import metadata
from sklearn import svm, pipeline, preprocessing as skpre
from sklearn import decomposition
#from sklearn import cross_validation -- Commented out, needs rewrite with model_selection module
import numpy as np
import pandas as pd
from joblib import Memory


sensors = dict(
    all=lambda x: [ch for ch in x if ch.startswith('M')],
    occipital=lambda x: [ch for ch in x if ch.startswith(
        'MLO') or ch.startswith('MRO')],
    posterior=lambda x: [ch for ch in x if ch.startswith(
        'MLP') or ch.startswith('MRP')],
    central=lambda x: [ch for ch in x if ch.startswith('MLC')
                       or ch.startswith('MRC') or ch.startswith('MZC')],
    frontal=lambda x: [ch for ch in x if ch.startswith('MLF')
                       or ch.startswith('MRF') or ch.startswith('MZF')],
    temporal=lambda x: [ch for ch in x if ch.startswith('MLT') or ch.startswith('MRT')])


def clf():
    return pipeline.Pipeline([
        ('scale', skpre.StandardScaler()),
        ('PCA', decomposition.PCA(n_components=.99)),
        ('SVM', svm.LinearSVC())])


def cv(x):
    return cross_validation.StratifiedShuffleSplit(x, n_iter=2, test_size=0.2)


def decode(classifier, data, labels, train_time, predict_times,
           cv=cross_validation.StratifiedKFold, collapse=np.mean,
           relabel_times=False,
           obs_average=None):
    '''
    Apply a classifier to data [epochs x channels x time] and predict labels with cross validation.
    Train classifier from data at data[:, :, train_time] and apply to all
    indices in predict_times. Indices are interpreted as an index into
    the data matrix.

    train_time can be a slice object to allow averaging across time or using
    time_sequences for prediction. In this case indexing data with train_time
    potentially results in a (n_epochs, n_channels, n_time) and the time
    dimension needs to be collapsed to obtain a (n_epochs, n_channels) matrix.
    How to do this can be controlled by the collapse keyword. If collapes=='reshape'
    data is coercd into  (n_epochs, n_channels*n_matrix), else collapse is applied
    to data like so:

        >>> X = collapse(data, axis=2)

    If train_time is a slice object, predict_times should be a list of slice objects
    and the test set gets the same treatment as the training set.

    Returns a vector of average accuracies for all indices in predict_idx.

    Parameters
    ----------
    classifier : sklearn classifier object
    data : np.array, (#trials, #features, #time)
    labels : np.array, (#trials,)
    train_time : int
    predict_times : iterable of ints
    '''
    assert len(labels) == data.shape[0]
    results = []

    # With only one label no decoding is possible.
    if len(np.unique(labels)) == 1:
        return pd.DataFrame({
            'fold': [np.nan],
            'train_time': [np.nan],
            'predict_time': [np.nan],
            'accuracy': [np.nan]
        })

    # Need at least two samples per class
    for label in np.unique(labels):
        if sum(labels == label) < 2:
            return pd.DataFrame({
                'fold': [np.nan],
                'train_time': [np.nan],
                'predict_time': [np.nan],
                'accuracy': [np.nan]
            })

    for i, (train_indices, test_indices) in enumerate(cv(labels)):
        np.random.shuffle(train_indices)
        fold = []
        clf = classifier()
        if len(np.unique(labels)) == 2:
            l1, l2 = np.unique(labels)
            l1 = train_indices[labels[train_indices] == l1]
            l2 = train_indices[labels[train_indices] == l2]
            if len(l1) > len(l2):
                l1 = l1[:len(l2)]
            else:
                l2 = l2[:len(l1)]
            assert not any([k in l2 for k in l1])
            train_indices = np.concatenate([l1, l2])
        train = data[train_indices, :, train_time]
        train_time_marker = train_time
        if len(train.shape) == 3:
            if collapse == 'reshape':
                train = train.copy().reshape(
                    (train.shape[0], np.prod(train.shape[1:])))
            else:
                train = collapse(train, axis=2)
            train_time_marker = train_time.stop - 1

        train_labels = labels[train_indices]
        if obs_average is not None:
            train, train_labels = obs_average(train, train_labels)

        clf = clf.fit(train, train_labels)

        for pt in predict_times:
            # print pt
            fold_result = {}
            test = data[test_indices, :, pt]
            if len(test.shape) == 3:
                if collapse == 'reshape':
                    test = test.reshape(
                        (test.shape[0], np.prod(test.shape[1:])))
                else:
                    test = collapse(test, axis=2)
                fold_result['predict_time'] = pt.stop - 1
            else:
                fold_result['predict_time'] = pt
            fold_result['train_time'] = train_time_marker

            if relabel_times is not False:
                fold_result['train_time'] = relabel_times[
                    fold_result['train_time']]
                fold_result['predict_time'] = relabel_times[
                    fold_result['predict_time']]
            # print 'Test:', test.shape
            test_labels = labels[test_indices]
            if obs_average is not None:
                test, test_labels = obs_average(test, test_labels)
            fold_result.update({
                'fold': i,
                'accuracy': clf.score(test, test_labels)})
            results.append(fold_result)
    return pd.DataFrame(results)


def observation_average(train, labels, N=5):
    '''
    Average X observations from each class into one observation.

    Works only for 2 classes at the moment.
    '''
    a, b = np.unique(labels)

    def foo(train, label, N):
        acc, lbl = [], []
        index = np.arange(train.shape[0])
        for idx in np.unique(index // N):
            acc.append(train[index == idx, :].mean(0))
            lbl.append(label[index == idx].mean())
        return np.vstack(acc), np.array(lbl)

    ta, la = foo(train[labels == a, :], labels[labels == a], N)
    tb, lb = foo(train[labels == b, :], labels[labels == b], N)
    return np.vstack([ta, tb]), np.concatenate((la, lb))


def generalization_matrix(epochs, labels, dt, slicelen=None,
                          classifier=clf, cv=cv, slices=False, baseline=None, obs_average=None):
    '''
    Get data for a generalization across time matrix.

    Parameters
    ----------
        epochs: mne.epochs object
    Epochs to use for decoding.
        labels : np.array (n_epochs,)
    Target labels to predict
        dt : int
    Time resolution of the decoding in ms (!).
        slices : False, 'reshape' or function
    Indicates how time dimension should be treated during decoding.
    False implies single time point decoding, reshape implies
    using time sequence data for decoding and function can be used
    to reduce time series data to single point.
    '''
    if slicelen is None:
        slicelen = dt
    data = epochs._data
    sfreq = epochs.info['sfreq']

    tlen = data.shape[-1] / (float(sfreq) / 1000.)
    nsteps = np.around(float(tlen) / dt)
    stepsize = max(1, int(data.shape[-1] / nsteps))
    steps = np.arange(0, data.shape[-1], stepsize)

    relabel_times = dict((k, v) for k, v in enumerate(epochs.times))

    if slices:
        slicelen = int(round(slicelen / (1000. / sfreq)))
        steps = [slice(s, s + slicelen)
                 for s in steps if (s + slicelen) < data.shape[-1]]
        decoder = lambda x: decode(clf, data, labels, x, steps, cv=cv,
                                   collapse=slices, relabel_times=relabel_times, obs_average=obs_average)
    else:
        decoder = lambda x: decode(clf, data, labels, x, steps, cv=cv,
                                   relabel_times=relabel_times, obs_average=obs_average)
    return pd.concat([decoder(tt) for tt in steps])


def apply_decoder(func, snum, epoch, label, channels=sensors['all'],
                  label_func=None):
    '''
    Apply a decoder function to epochs from a subject and decode 'label'.

    Parameters
    ----------
        func: function object
    A function that performs the desired decoding. It needs to take two arguments
    that the epoch object and labels to use for the decoing. E.g.:

        >>> func = lambda x,y: generalization_matrix(x, y, 10)

        snum: int
    Subject number to indicate which data to load.
        epoch: str
    One of 'stimulus', 'response', or 'feedback'
        label: str
    Which column in the metadata to use for decoding. Labels will recoded to
    0-(num_classes-1).
    '''
    s, m = preprocessing.get_epochs_for_subject(
        snum, epoch)  # This will cache.
    s = s.pick_channels(channels(s.ch_names))
    times = s.times
    # Add confidence labels
    idx = m.response == 1
    r1c = m.confidence.copy()
    r1c[~idx] = np.nan
    rm1c = m.confidence.copy()
    rm1c[idx] = np.nan
    m.loc[:, 'conf_rm1'] = rm1c
    m.loc[:, 'conf_r1'] = r1c

    # Drop nan labels
    labels = m.loc[:, label]
    if label_func is not None:

        labels = label_func(labels)
        nan_loc = m.index[np.isnan(labels)]
        use_loc = m.index[~np.isnan(labels)]
    else:
        nan_loc = m.index[np.isnan(m.loc[:, label])]
        use_loc = m.index[~np.isnan(m.loc[:, label])]

    m = m.drop(nan_loc)
    s = s[list(use_loc.astype(str))]

    # Sort order index to align epochs with labels.
    m = m.loc[s.events[:, 2]]
    if not all(s.events[:, 2] == m.index.values):
        raise RuntimeError('Indices of epochs and meta do not match! Task: ' +
                           str(snum) + ' ' + epoch + ' ' + label)
    # Recode labels to 0-(n-1)

    labels = m.loc[:, label]
    if label_func is not None:
        labels = label_func(labels)
    labels = skpre.LabelEncoder().fit(labels).transform(labels)
    return func(s, labels)


def to4d(a):
    trial = toindex(np.unique(a.index.get_level_values('trial')))
    channel = toindex(np.unique(a.index.get_level_values('channel')))
    freqs = toindex(np.unique(a.index.get_level_values('freq')))
    times = toindex(np.unique(a.columns.values))
    out = np.empty((len(trial), len(channel), len(freqs), a.shape[1]))
    for row in a.itertuples():
        t, c, f, s = row[0]
        out[trial[t], channel[c], freqs[f], :] = row[1:]
    return out, (trial, channel, freqs, times)


def toindex(x):
    return dict((v, k) for k, v in enumerate(x))


def tfr_apply_decoder(func, snum, epoch, label,
                      channels=sensors['all'], freq=(0, 150),
                      label_func=lambda x: x, time=(-0.5, 1.25), baseline=(-0.5, 0)):
    '''
    Apply a decoder function to TFR from a subject and decode 'label'.

    Parameters
    ----------
        See apply_decoder
    '''
    s, m = tfr_analysis.get_subject(snum, freq, channels, time[0],  time[1],
                                    epoch=epoch)
    s = s.groupby(level='channel').apply(
        lambda x: tfr_analysis.avg_baseline(x, baseline=baseline))

    m = m.loc[~np.isnan(label_func(m[label])), :].sort_index()
    label_index = list(np.sort(m.index.values))
    m = m.loc[label_index, :]

    s = s.sort_index()
    s = s.loc[(label_index,), :]
    #s = s.sort_index()

    s, indices = to4d(s)
    hashloc = toindex(m.index.values)
    lval = m[label].values

    labels = [lval[indices[0][i]] for i in m.index.values]
    # Assert correct labelling
    rs = dict((v, k) for k, v in list(indices[0].items()))

    assert(
        all(
            [label_func(labels[i]) == label_func(m.loc[rs[i], label])
                for i in np.arange(len(labels))]))
    if label_func is not None:
        labels = label_func(labels)

    labels = skpre.LabelEncoder().fit(labels).transform(labels)
    #s[labels==1, :, :, :] = np.random.randn(*s[labels==1, :, :, :].shape) + 1
    #s[labels==0, :, :, :] = np.random.randn(*s[labels==0, :, :, :].shape) - 1
    # return s, indices, labels
    return func((s, indices), labels)


def tfr_generalization_matrix(epochs, labels,
                              classifier=clf, cv=cv, dt=1, slices='reshape', baseline=None,
                              relabel_times=False):
    '''
    Get data for a generalization across time matrix.

    epochs is a tuple: 4D array (trial, channel, freq, time), indexers
        indexers are dicts that map keys to index.
    '''

    indexers = epochs[1]
    data = epochs[0]
    h, c, f, t = data.shape

    data = data.reshape((h, c * f, t))
    steps = [slice(t, t + dt) for t in range(data.shape[-1] - dt)]
    print(data.shape)
    print(steps)
    relabel_times = dict((v, k) for k, v in list(indexers[-1].items()))
    decoder = lambda x: decode(clf, data, labels, x, [x], cv=cv,
                               relabel_times=relabel_times)
    return pd.concat([decoder(tt) for tt in steps])
