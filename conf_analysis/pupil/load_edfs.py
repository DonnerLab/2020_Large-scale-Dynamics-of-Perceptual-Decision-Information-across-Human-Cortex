import collections
from scipy.io import loadmat
import pandas as pd
from pyedfread import edf
import numpy as np
import pylab
from . import pupil


def listfiles(dir):
    import glob, os, time
    edffiles = glob.glob(os.path.join(dir, '*.edf'))
    edffiles = [k for k in edffiles if 'localizer' not in k]
    matfiles = glob.glob(os.path.join(dir, '*.mat'))
    edfdata = {}
    matdata = {}
    subs = []
    for f in edffiles:
        if 'localizer' in f:
            continue
        sub, d = f.replace('.edf', '').split('/')[-1].split('_')
        edfdata[time.strptime(d, '%Y%m%dT%H%M%S')] = f
        subs.append(sub)
    for f in matfiles:
        if 'localizer' in f or 'quest' in f:
            continue
        sub, d, ll = f.replace('.mat', '').split('/')[-1].split('_')
        matdata[time.strptime(d, '%d-%b-%Y %H:%M:%S')] = f
        subs.append(sub)
    if not (len(np.unique(subs))==1):
        print(np.unique(subs))
        raise RuntimeError('Files from more than one sub in folder')
    return edfdata, matdata, sub



###
###  Trig A
###

def expand_trigA(events, messages, field, align_key='trial', default=np.nan):
    '''
    Expand contrast for files with Trig A version
    '''
    assert(len(np.unique(events[align_key])) == 1)
    trial = events[align_key].iloc[0]
    messages = messages[messages[align_key] == trial]
    try:
        con = messages[field].iloc[0]
        con_on = messages[field + '_time'].iloc[0]
    except IndexError:
        events[field] =  default + np.zeros(len(events))
        return events

    trial_begin = events.index.values[0]
    contrast = default + np.zeros(len(events))
    try:
        for (start, end), value in zip(list(zip(con_on[:-1], con_on[1:])), con):
            contrast[start-trial_begin:end-trial_begin] = value
        contrast[end-trial_begin:end-trial_begin+100] = con[-1]
    except TypeError:
        pass
    events[field] = contrast

    return events


def load_edf(filename):
    '''
    Loads one EDF file and returns a clean DataFrame.
    '''
    events, messages = edf.pread(
        filename,
        properties_filter=['gx', 'gy', 'pa', 'sttime', 'start', 'type', 'gavx', 'gavy'],
        filter='all')
    #events = edf.trials2events(events, messages)
    #events['stime'] = pd.to_datetime(events.sample_time, unit='ms')
    events['type'] = events.type=='fixation'
    if all((events.right_pa == -32768) | np.isnan(events.right_pa)):
        del events['right_pa']
        events['pa'] = events.left_pa
        del events['left_pa']
    elif all((events.left_pa == -32768) | np.isnan(events.left_pa)):
        del events['left_pa']
        events['pa'] = events.right_pa
        del events['right_pa']
    else:
        raise RuntimeError('Recorded both eyes? So unusual that I\'ll stop here')

    # In some cases the decision variable still contains ['second', 'conf', 'high'], 21.0 -> Fix this
    # In these cases the decision_time variable has 2 time stamps as well...
    if messages.decision.dtype == np.dtype(object):
        messages['decision'] = np.array([x[-1] if isinstance(x, collections.Sequence) else x for x in messages.decision.values])
        messages['decision_time'] = np.array([x[-1] if isinstance(x, collections.Sequence) else x for x in messages.decision_time.values])

    return events, messages


def preprocess_trigA(events, messages):
    Ti = np.nanmean(np.diff(events.sample_time))
    down_factor = 10
    if abs((Ti-1))>abs((Ti-2)):
        down_factor = 5
    print(Ti, down_factor)
    events = events.set_index('sample_time')
    events = events.groupby('trial').apply(lambda x: expand_trigA(x, messages, 'conrast', default=0))
    down_factor = 10
    if abs((Ti-1))>abs((Ti-2)):
        down_factor = 5
    events = pupil.decimate(events, down_factor)

    events = events[~np.isnan(events.pa)]
    con_on = [c.conrast_time.values[0][0] for name, c in messages.groupby('trial')]
    messages['contrast_on'] = con_on
    join_msg_trigA(events, messages)
    below, filt, above = pupil.filter_pupil(events.pa, 100)
    events['pafilt'] = filt
    events['palow'] = below
    events['pahigh'] = above
    return events, messages


def join_msg_trigA(events, messages):
    # Join messages into events. How to do this depends a bit on the semantics of the message
    paired_fields = ['decision', 'feedback']
    time_index = events.index.values
    for field in paired_fields:
        events[field] = np.zeros(events.pa.values.shape)
        for t,v in zip(messages[field + '_time'].values, messages[field].values):
            idx = np.argmin(np.abs(time_index-t))
            if field == 'feedback' and v == 0:
                v=-1
            events[field].iloc[idx] = v


###
### Trig B
###


def expand_trigB(events, messages, field, align_key='trial', align_time='trial_time', default=np.nan):
    '''
    --- Missing ---
    '''
    assert(len(np.unique(events[align_key])) == 1)
    trial = events[align_key].iloc[0]
    messages = messages[messages[align_key] == trial]
    try:
        con = messages[field].iloc[0]
        con_on = messages[align_time].iloc[0]
        trial_begin = events.index.values[0]
        events[field] = default + np.zeros(len(events))
        for (start, end), value in zip(list(zip(con_on[:-1], con_on[1:])), con):
            events[field].loc[start:end] = value
        events[field].loc[end:end+100] = con[-1]
    except (IndexError, TypeError):
        events[field] =  default + np.zeros(len(events))

    return events


def load_behavior_trigB(behavioral):
    def unbox_messages(current):
        for key in list(current.keys()):
            try:
                if len(current[key])==1:
                    current[key] = current[key][0]
            except TypeError:
                pass
        return current
    # Also load behavioral results file to match with contrast levels shown
    behavioral = loadmat(behavioral)['session_struct']['results'][0,0]
    d = []
    fields = list(behavioral.dtype.fields.keys())
    for trial in range(behavioral.shape[1]):
        d.append({})
        for field in fields:
            d[-1][field] = behavioral[0, trial][field].ravel()
        d[-1]['trial'] = trial+1
        d[-1] = unbox_messages(d[-1])
    return pd.DataFrame(d)


def join_edf_and_behavior(messages, behavior):
    return messages.set_index('trial').join(behavior.set_index('trial')).reset_index()


def preprocess_trigB(events, messages, behavior=None):
    if behavior is not None:
        messages = join_edf_and_behavior(messages, behavior)
    Ti = np.nanmean(np.diff(events.sample_time))
    events = events.set_index('sample_time')
    events = events.groupby('trial').apply(lambda x: expand_trigB(x, messages, 'contrast_probe',
                                                            align_time='con_change_time', default=0))
    down_factor = 10
    if abs((Ti-1))>abs((Ti-2)):
        down_factor = 5
    events = pupil.decimate(events, down_factor)
    events = events[~np.isnan(events.pa)]
    #con_on = [c.conrast_time.values[0][0] for name, c in messages.groupby('trial')]
    #messages['contrast_on'] = con_on
    join_msg_trigB(events, messages)
    below, filt, above = pupil.filter_pupil(events.pa, 100)
    events['pafilt'] = filt
    events['palow'] = below
    events['pahigh'] = above
    return events, messages


def join_msg_trigB(events, messages):
    # Join messages into events. How to do this depends a bit on the semantics of the message
    paired_fields = ['decision', 'feedback']
    time_index = events.index.values
    for field in paired_fields:
        events[field] = np.zeros(events.pa.values.shape)
        for t,v in zip(messages[field + '_time'].values, messages[field].values):
            idx = np.argmin(np.abs(time_index-t))
            if field == 'feedback' and v == 0:
                v=-1
            events[field].iloc[idx] = v


def get_data(edf, behavior):
    events, messages = load_edf(edf)

    if 'con_change' in messages.columns.values.tolist():
        behavior = load_behavior_trigB(behavior)
        events, messages = preprocess_trigB(events, messages, behavior)
        events['contrast'] = events['contrast_probe']
        del events['contrast_probe']
        messages['contrast'] = messages['contrast_probe']
        messages['contrast_time'] = messages['con_change_time']
        messages.drop(['Noise_sigma', 'Noise_sigma_time', 'Stim_correct',
                  'Stim_correct_time', 'stim_off', 'stim_off_time',
                  'contrast_probe', 'correct', 'choice_rt', 'confidence',
                  'contrast_ref', 'expand', 'repeat', 'response', 'session', 'side',
                  'stim_onset', 'stim_onset_time', 'noise_sigma', 'repeated_stim',
                  'con_change', 'con_change_time'], 1, inplace=True)
    else:
        events, messages = preprocess_trigA(events, messages)
        events['contrast'] = events['conrast']
        #events['contrast_time'] = events['conrast_time']
        events.drop(['conrast'], 1, inplace=True)
        messages['contrast'] = messages['conrast']
        messages['contrast_time'] = messages['conrast_time']
        messages.drop(['conrast', 'conrast_time'], 1, inplace=True)
    sfname = edf.split('/')[-1].split('.')[0]
    events['sfname'] = sfname
    return events, messages


def get_sub_df(sub):
    import datadef
    es = []
    msgs = []
    keys = []
    sub_files = datadef.sb2fname[sub]
    for session, sdata in sub_files.items():
        for block, (edf, matfile) in sdata.items():
            try:
                print('\nProcessing S%i, B%i: '%(session, block), edf)
                events, messages = get_data(edf, matfile)
            except IOError:
                print('\nS%i, B%i is corrupt'%(session, block), edf)
                continue
            events['block'] = block
            events['session'] = session
            events['subject'] = sub
            messages['block'] = block
            messages['session'] = session
            messages['subject'] = sub
            es.append(events)
            msgs.append(messages)
            keys.append((session, block))
    events = pd.concat(es, keys=keys)
    messages = pd.concat(msgs, keys=keys)
    return events, messages

def save_sub(sub, path='temp_data'):
    events, messages = get_sub_df(sub)
    import pickle, os, gzip
    file = gzip.open(os.path.join(path, sub+'.pickle.gzip'), 'w')
    pickle.dump({'events':events, 'messages':messages}, file, protocol=2)
    file.close()

if __name__ == '__main__':
    import sys
    import locale
    subject = int(sys.argv[1])
    locale.setlocale(locale.LC_ALL, 'en_US')
    save_sub('S%02i'%subject)
