import numpy as np
import mne
from pymeg import preprocessing, tfr, artifacts
from sklearn import cluster, neighbors
from sklearn.metrics import pairwise
import glob
from joblib import Memory
from conf_analysis.behavior import metadata
from conf_analysis.meg import tfr_analysis as ta
memory = Memory(cachedir=metadata.cachedir, verbose=0)

try:
    params = tfr.params_from_json(
        '/home/nwilming/conf_analysis/all_tfr150_parameters.json')
except IOError:
    params = None


def select(snum, reject=dict(mag=4e-12), params=params):
    import pylab as plt
    import pickle

    power, power_std = get_localizer_power(
        snum, reject=reject, params=params).crop(tmin=-0.4, tmax=0.75)
    clusters, ch_names, f_roi, freq_resp, dists = cluster_sensors(power,
                                                                  n_clusters=10, freq=slice(20, 80))
    pickle.dump(
        (clusters, ch_names, f_roi, freq_resp, dists),
        open('/home/nwilming/conf_analysis/localizer_results/lr_%i_gamma.pickle' % snum, 'w'))
    plt.figure(figsize=(15, 30))
    make_selection_plot(power, clusters, ch_names, f_roi, freq_resp, dists)
    plt.savefig('/home/nwilming/conf_analysis/localizer_results/lr_%i_plot.png' %
                snum, bbox_inches="tight")


def select_by_gamma(snum, reject=dict(mag=4e-12), params=params, n=10):
    import pylab as plt
    import pickle
    n_sensors = 1
    power, power_std = get_localizer_power(
        snum, reject=reject, params=params, sensors=['occipital', 'posterior'])
    chn = np.array(power.ch_names)
    power = power.apply_baseline((-0.2, 0), mode='zscore')
    power_std = power.data.std(0)
    averages = power.average()
    averages.crop(tmin=0.2, tmax=0.75)
    id_time = np.in1d(power.times, averages.times)
    power_std = power_std[:, :, id_time]
    # Compute average gamma band power and order by it
    id_f = (35 < averages.freqs) & (averages.freqs < 60)
    a = averages.data[:, id_f, :].mean(axis=(1, 2))
    v = power_std[:, id_f, :].mean(axis=(1, 2))
    s = a / v

    ordering = np.argsort(s)
    mapping = dict((chn[k], k) for k in ordering if chn[
                   k] in ta.sensors['occipital'](chn))
    ordering = [channel for channel in ordering if chn[
        channel] in list(mapping.keys())]

    pos = np.asarray([ch['loc'][:3] for ch in power.info['chs']])
    N = neighbors.kneighbors_graph(
        pos, n_neighbors=n_sensors, include_self=True)

    clusters = [np.where(N[i].toarray())[1] for i in ordering[-n:]]
    ch_names, f_roi, freq_resp, dists = [], [], [], []
    for cluster in clusters:
        fresp = (averages.data[cluster, :, :] /
                 power_std[cluster, :, :]).mean(0)
        f_roi.append(fresp.mean(1))
        freq_resp.append(fresp.mean(1))
        ch_names.append(np.array(averages.ch_names)[cluster])
    dists = np.ones((269, 269))
    pickle.dump(
        (clusters, ch_names, f_roi, freq_resp, dists),
        open('/home/nwilming/conf_analysis/localizer_results/lrbg_%i_gamma.pickle' % snum, 'w'))
    plt.figure(figsize=(15, 30))
    make_selection_plot2(averages, clusters, ch_names,
                         f_roi, freq_resp, dists, chn)
    plt.savefig('/home/nwilming/conf_analysis/localizer_results/lrbg_%i_plot.png' %
                snum, bbox_inches="tight")


@memory.cache
def get_localizer_epochs(filename, reject=dict(mag=4e-12)):
    import locale
    locale.setlocale(locale.LC_ALL, "en_US")
    raw = mne.io.read_raw_ctf(filename, system_clock='ignore')
    sf = float(raw.info['sfreq'])
    mapping = {50: ('con_change', 0), 64: ('stim_onset', 0),
               160: ('start', 0), 161: ('end', 0)}
    meta, timing = preprocessing.get_meta(raw, mapping, {}, 160, 161)
    if len(meta) == 0:
        return None
    tmin, tmax = (timing.min().min() / sf) - \
        5, (max(timing.max().max()) / sf) + 5
    raw = raw.crop(tmin=max(0, tmin),
                   tmax=min(tmax, raw.times[-1]))
    raw.load_data()
    raw.notch_filter(np.arange(50, 251, 50), n_jobs=4)
    meta, timing = preprocessing.get_meta(raw, mapping, {}, 160, 161)

    l = len(timing)
    events = np.vstack([timing.stim_onset_time, [0] * l, [1] * l]).T
    e = mne.Epochs(raw, events=events.astype(int),
                   tmin=-1.25, tmax=1.75, reject=reject,
                   )
    del raw
    return e


def get_localizer(snum, reject=dict(mag=4e-12)):
    files = glob.glob('/home/nwilming/conf_meg/raw/s%02i-*.ds' % snum)
    files += glob.glob('/home/nwilming/conf_meg/raw/S%02i-*.ds' % snum)
    if snum == 14:
        files += glob.glob('/home/nwilming/conf_meg/raw/%02i-*.ds' % snum)
    epochs = [get_localizer_epochs(f) for f in files]
    epochs = [e for e in epochs if e is not None]
    dt = epochs[0].info['dev_head_t']
    for e in epochs:
        e.info['dev_head_t'] = dt
        e.reject = reject
        e.load_data()
    return mne.concatenate_epochs([e for e in epochs if len(e) > 0])


@memory.cache
def get_localizer_power(snum, reject=dict(mag=4e-12), params=params, sensors=None):
    print('Getting localizer')
    epochs = get_localizer(snum, reject=reject)
    x, y = eye_voltage2gaze(epochs)
    arts = detect_eyeartifact(epochs.times, x, y)
    epochs.drop(arts)
    if sensors is not None:
        print('Kicking sensors')
        allowed_sensors = []
        for s in sensors:
            allowed_sensors.extend(ta.sensors[s](epochs.ch_names))
        epochs.drop_channels(
            [ch for ch in epochs.ch_names if ch not in allowed_sensors])
    print('Number of channels in epochs:', len(epochs.ch_names))
    print('Dropping %i epochs because of eye movements' % sum(arts))
    epochs.pick_channels([ch for ch in epochs.ch_names if ch.startswith('M')])
    epochs.resample(400, n_jobs=4)
    print('Doing power calculation')
    power = get_power(epochs, params=params)
    return power, power.data.std(0)


def get_power(epochs, params=params):
    return mne.time_frequency.tfr_multitaper(epochs,
                                             params['foi'], params['cycles'], average=False, return_itc=False, n_jobs=4)


def get_sensor_selection(power, freqs=(50, 110),
                         period=(0.05, 0.3),  cutoffs=[2.5, 5, 7.5, 10]):

    # Discard first and last bits of data
    id_use = (-0.5 < power.times) & (power.times < 1.1)
    id_freqs = (freqs[0] < power.freqs) & (power.freqs < freqs[1])
    data = power.data[:, id_freqs, :][:, :, id_use]
    times = power.times[id_use]
    id_base = (-0.25 < times) & (times < 0)
    base_mean = data[:, :, id_base].mean(2)[:, :, np.newaxis]
    base_std = data[:, :, id_base].std(2)[:, :, np.newaxis]
    data = (data - base_mean) / base_std
    data = data.mean(1)
    sensor_id, sensor_names = [], []
    start, end = period
    id_t = (start <= times) & (times < end)
    for cutoff in cutoffs:
        sensor_id.append(np.where(abs(data[:, id_t].mean(1)) > cutoff)[0])
        sensor_names.append([power.ch_names[sid] for sid in sensor_id[-1]])
    return data, times, sensor_id, sensor_names


def cluster_sensors(power, n_clusters=5, freq=slice(5, 200)):
    '''
    Use hierarchical agglomerative clustering (ward) to select channels.
    '''
    freqs = power.freqs
    pos = np.asarray([ch['loc'][:3] for ch in power.info['chs']])
    averages = (power.copy()
                     .apply_baseline((-0.2, 0), mode='zscore')
                     .crop(tmin=0.00, tmax=1))

    id_freqs = (freq.start < freqs) & (freqs < freq.stop)
    data = averages.data[:, id_freqs, :]
    dims = data.shape
    dists = pairwise.pairwise_distances(
        data.reshape(dims[0], dims[1] * dims[2]), metric='l2')

    N = neighbors.kneighbors_graph(pos, n_neighbors=4)
    cl = cluster.AgglomerativeClustering(
        n_clusters=n_clusters,  linkage='average', connectivity=N)
    #cl = cluster.DBSCAN(eps=0.2)
    #cl = cluster.KMeans(n_clusters=n_clusters)
    cl.fit(dists)
    labels = cl.labels_
    clusters = [np.where(cl.labels_ == k)[0] for k in np.unique(cl.labels_)]
    freq_resp = [averages.data.mean(-1)[c, :].mean(0) for c in clusters]
    f_roi = [freqs[np.argmax(fr)] for fr in freq_resp]
    ch_names = [[power.ch_names[k] for k in clt] for clt in clusters]
    return clusters, ch_names, f_roi, freq_resp, dists


def make_selection_plot(power, clusters, ch_names, f_roi, freq_resp, dists):
    import pylab as plt
    import matplotlib.gridspec as gridspec
    averages = (power.copy()
                     .apply_baseline((-0.2, 0), mode='zscore')
                     .crop(tmin=0.00, tmax=1))
    nc = len(clusters)
    gs = gridspec.GridSpec(nc + 1, 2)
    plt.subplot(gs[0, 0])
    plot_sensors(power.info, ch_groups=clusters)
    lines = plt.gca().get_lines()
    for line in lines:
        line.set_markeredgecolor('none')
    plt.subplot(gs[0, 1])
    order = list(np.concatenate(clusters))
    plt.imshow(dists[order][:, order], aspect='auto', interpolation='none')
    for i, (channels, fr, pf) in enumerate(zip(clusters, freq_resp, f_roi)):
        plt.subplot(gs[i + 1, 0])
        plt.title('Cluster %i' % i)
        plot_tfr(averages, channels)
        plt.subplot(gs[i + 1, 1])
        plt.plot(power.freqs, fr)
        plt.axvline(pf, color='r')
    plt.tight_layout()


def make_selection_plot2(power, clusters, ch_names, f_roi, freq_resp, dists, all_channels):
    import pylab as plt
    import matplotlib.gridspec as gridspec
    averages = power
    nc = len(clusters)
    gs = gridspec.GridSpec(nc + 1, 3)
    plt.subplot(gs[0, 0])
    plot_sensors(power.info, ch_groups=clusters)
    lines = plt.gca().get_lines()
    for line in lines:
        line.set_markeredgecolor('none')
    plt.subplot(gs[0, 1])
    order = list(np.concatenate(clusters))
    plt.imshow(dists[order][:, order], aspect='auto', interpolation='none')
    plt.title('N=%i' % power.nave)

    for i, (channels, fr, pf) in enumerate(zip(clusters, freq_resp, f_roi)):
        plt.subplot(gs[i + 1, 0])
        groups = [channels, list(
            set(np.arange(len(all_channels))) - set(channels))]
        plot_sensors(power.info, ch_groups=groups)
        plt.subplot(gs[i + 1, 1])
        plt.title('Cluster %i' % i)
        plot_tfr(averages, channels)
        plt.subplot(gs[i + 1, 2])
        plt.plot(power.freqs, fr)
        #plt.axvline(pf, color='r')
    plt.tight_layout()


def plot_tfr(averages, picks, cmap='RdBu_r', vmin=-12, vmax=12,
             mark_max=True, max_tmin=0.2):
    vext = abs(vmin) if abs(vmin) > vmax else vmax
    import pylab as plt
    freq = averages.freqs
    times = averages.times
    ratio = freq[1:] / freq[:-1]
    log_freqs = np.concatenate(
        [[freq[0] / ratio[0]], freq, [freq[-1] * ratio[0]]])
    freq_lims = np.sqrt(log_freqs[:-1] * log_freqs[1:])

    time_diff = np.diff(times) / 2. if len(times) > 1 else [0.0005]
    time_lims = np.concatenate([[times[0] - time_diff[0]], times[:-1] +
                                time_diff, [times[-1] + time_diff[-1]]])

    time_mesh, freq_mesh = np.meshgrid(time_lims, freq_lims)
    data = averages.data[picks, :].mean(0)

    plt.pcolormesh(time_mesh, freq_mesh, data,
                   cmap=cmap, vmin=-vext, vmax=vext)
    if mark_max:
        idx = times > max_tmin
        mx = freq[np.argmax(data, 0)]
        plt.plot(times[idx], mx[idx], 'k')
        mmx = np.around(np.mean(mx[idx]), 2)
        plt.text(max_tmin, mmx+10, 'Mean Hz: %3.2f' % mmx)

    plt.xlim([time_lims[0], time_lims[-1]])
    plt.ylim([freq_lims[0], freq_lims[-1]])


def eye_voltage2gaze(epochs, ranges=(-5, 5), screen_x=(0, 1920),
                     screen_y=(0, 1080),
                     ch_mapping={'x': 'UADC002-3705', 'y': 'UADC003-3705', 'p': 'UADC004-3705'}):
    '''
    Convert analog output of EyeLink 1000+ to gaze coordinates.
    '''
    minvoltage, maxvoltage = ranges
    maxrange, minrange = 1., 0.
    screenright, screenleft = screen_x
    screenbottom, screentop = screen_y
    idx = np.where(np.array(epochs.ch_names) == ch_mapping['x'])[0][0]
    raw = epochs.get_data()[:, idx, :]

    R = (raw - minvoltage) / (maxvoltage - minvoltage)
    S = R * (maxrange - minrange) + minrange
    x = S * (screenright - screenleft + 1) + screenleft

    idy = np.where(np.array(epochs.ch_names) == ch_mapping['y'])[0][0]
    raw = epochs.get_data()[:, idy, :]
    R = (raw - minvoltage) / (maxvoltage - minvoltage)
    S = R * (maxrange - minrange) + minrange
    y = S * (screenbottom - screentop + 1) + screentop

    return x, y


def detect_eyeartifact(times, xx, yy, time=(-.2, 0.5), Hz=1000, ppd=45.):
    arts = []
    for x, y in zip(xx, yy):
        idt = (time[0] < times) & (times < time[1])
        saccades = artifacts.saccade_detection(
            x[idt] / ppd, y[idt] / ppd, Hz=Hz)
        if len(saccades) > 0:
            arts.append(True)
        else:
            arts.append(False)
    return arts


def plot_sensors(info, ch_groups=None):
    import pylab as plt
    import seaborn as sns
    from mne.viz.utils import channel_indices_by_type, _contains_ch_type, _auto_topomap_coords
    ch_indices = channel_indices_by_type(info)

    picks = ch_indices['mag']
    pos = np.asarray([ch['loc'][:3] for ch in info['chs']])[picks]
    ch_names = np.array(info['ch_names'])[picks]
    pos = _auto_topomap_coords(info, picks, True, to_sphere=True)
    if ch_groups is not None:
        colors = sns.color_palette("hls", len(ch_groups))
        colors = sns.color_palette("Paired", len(ch_groups))
        for i, group in enumerate(ch_groups):
            plt.plot(pos[group, 0], pos[group, 1], 'o',
                     label='C %i' % i, mec='none', color=colors[i])
    plt.legend(ncol=5, loc=0, bbox_to_anchor=(0., 1.02, 1., .102),
               mode="expand", borderaxespad=0.)
    plt.xticks([])
    plt.yticks([])
    sns.despine(left=True, bottom=True)
