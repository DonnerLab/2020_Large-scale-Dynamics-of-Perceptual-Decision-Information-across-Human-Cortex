'''
Keep track of all the subject data
'''
from numpy import *
import numpy as np
import os
import socket
import pandas as pd

if socket.gethostname().startswith('node'):
    home = '/home/nwilming/'
    project = '/home/nwilming/conf_analysis/'
    raw_path = '/home/nwilming/conf_meg/raw/'
    behavioral_path = '/home/nwilming/conf_data/'
    if (('RRZ_TMPDIR' in list(os.environ.keys()))
            or ('RRZ_LOCAL_TMPDIR' in list(os.environ.keys()))):
        cachedir = '/work/faty014/cache_dir'
        preprocessed = '/work/faty014/MEG/preprocessed'
        sr_labeled = '/work/faty014/MEG/sr_labeled/'
        sraggregates = '/work/faty014/MEG/sr_labeled/aggregates'
    else:
        cachedir = '/home/nwilming/conf_data/cache/'
        preprocessed = '/home/nwilming/conf_meg/'
        sr_labeled = '/home/nwilming/conf_meg/sr_labeled/'
        sraggregates = '/home/nwilming/conf_meg/sr_labeled/aggregates'
elif 'lisa.surfsara' in socket.gethostname():
    home = '/home/nwilming/'
    project = '/home/nwilming/conf_analysis/'
    cachedir = '/nfs/nwilming'
    sraggregates = '/nfs/nwilming/MEG/sr_labeled'
    preprocessed = '/nfs/nwilming/MEG/preprocessed/'
elif 'orthosie' in socket.gethostname():
    home = '/net/home/student/n/nwilming'
    project = '/net/home/student/n/nwilming/conf_analysis/'
    cachedir = '/net/home/student/n/nwilming/pymeg_cache'
    sraggregates = '/nfs/nwilming/MEG/sr_labeled'
    preprocessed = '/nfs/nwilming/MEG/preprocessed/'
else:
    home = '/Users/nwilming/'
    project = '/Users/nwilming/u/conf_analysis/'
    raw_path = '/Volumes/dump/conf_data/raw/'
    preprocessed = '/Volumes/dump/conf_data/'
    cachedir = '/Users/nwilming/u/conf_analysis/cache/'
    behavioral_path = '/Users/nwilming/u/conf_data/'


data_files = {'S01': ['s01-01_Confidence_20151208_02.ds',
                      's01-02_Confidence_20151210_02.ds',
                      's01-03_Confidence_20151216_01.ds',
                      's01-04_Confidence_20151217_02.ds'],
              'S02': ['s02-01_Confidence_20151208_02.ds',
                      's02-02_Confidence_20151210_02.ds',
                      's02-03_Confidence_20151211_01.ds',
                      's02-4_Confidence_20151215_02.ds'],
              'S03': ['s03-01_Confidence_20151208_01.ds',
                      's03-02_Confidence_20151215_02.ds',
                      's03-03_Confidence_20151216_02.ds',
                      's03-04_Confidence_20151217_02.ds'],
              'S04': ['s04-01_Confidence_20151210_02.ds',
                      's04-02_Confidence_20151211_02.ds',
                      's04-03_Confidence_20151215_02.ds',
                      's04-04_Confidence_20151217_02.ds'],
              'S05': ['s05-01_Confidence_20151210_02.ds',
                      's05-02_Confidence_20151211_02.ds',
                      's05-03_Confidence_20151216_02.ds',
                      's05-04_Confidence_20151217_02.ds'],
              'S06': ['s06-01_Confidence_20151215_02.ds',
                      's06-02_Confidence_20151216_02.ds',
                      's06-03_Confidence_20151217_02.ds',
                      's06-04_Confidence_20151218_04.ds'],
              'S07': ['S07-01_Confidence_20160216_01.ds',
                      'S07-2_Confidence_20160224_01.ds',
                      'S07-3_Confidence_20160302_01.ds',
                      'S07-4_Confidence_20160303_01.ds'],
              'S08': ['S08-01_Confidence_20160216_01.ds',
                      'S08-2_Confidence_20160225_01.ds',
                      'S08-3_Confidence_20160229_01.ds',
                      'S08-4_Confidence_20160302_01.ds'],
              'S09': ['S09-01_Confidence_20160216_01.ds',
                      'S09-2_Confidence_20160223_01.ds',
                      'S09-3_Confidence_20160226_01.ds',
                      'S09-4_Confidence_20160303_01.ds'],
              'S10': ['S10-01_Confidence_20160223_01.ds',
                      'S10-2_Confidence_20160224_01.ds',
                      'S10-3_Confidence_20160301_02.ds',
                      'S10-4_Confidence_20160302_01.ds'],
              'S11': ['S11-1_Confidence_20160223_01.ds',
                      'S11-2_Confidence_20160224_01.ds',
                      'S11-3_Confidence_20160225_01.ds',
                      'S11-4_Confidence_20160226_01.ds'],
              'S12': ['S12-1_Confidence_20160225_01.ds',
                      'S12-2_Confidence_20160226_01.ds',
                      'S12-3_Confidence_20160303_01.ds',
                      'S12-4_Confidence_20160304_01.ds'],
              'S13': ['S13-1_Confidence_20160301_01.ds',
                      'S13-2_Confidence_20160302_01.ds',
                      'S13-3_Confidence_20160303_01.ds',
                      'S13-4_Confidence_20160304_01.ds'],
              'S14': ['14-1_Confidence_20160301_01.ds',
                      'S14-2_Confidence_20160302_01.ds',
                      'S14-3_Confidence_20160303_01.ds',
                      'S14-4_Confidence_20160304_01.ds'],
              'S15': ['S15-1_Confidence_20160421_01.ds',
                      'S15-2_Confidence_20160425_01.ds',
                      'S15-3_Confidence_20160427_01.ds',
                      'S15-4_Confidence_20160428_01.ds']
              }

file_type_map = {'fif': '-epo.fif.gz',
                 'artifacts': '.artifact_def', 'meta': '.meta'}


def get_raw_filename(snum, session):
    return os.path.join(raw_path,
                        data_files['S%02i' % snum][session])


def get_epoch_filename(snum, session, block, period, data_type):
    '''
    Return filename for a epoch object.

    Parameters:
        snum, session, block : int
    Subject, session and block of desired epoch object.
        period : str
    One of 'response', 'stimulus', 'feedback'
        type : str
    One of 'fif', 'artifcats', 'meta'
    '''
    assert(data_type in list(file_type_map.keys()))
    path = os.path.join(preprocessed, 'S%i' % snum)
    if period is None:
        fname = 'SUB%i_S%i_B%i' % (
            snum, session, block) + file_type_map[data_type]
    else:
        fname = 'SUB%i_S%i_B%i_%s' % (
            snum, session, block, period) + file_type_map[data_type]
    fname = os.path.join(path, fname)
    return fname


def define_blocks(events):
    '''
    Parse block structure from events in MEG files.

    Aggresively tries to fix introduced by recording crashes and late recording
    starts.

    Beware: The trial number and block numbers will not yet match to the behavioral
    data.

    TODO: Doc.
    '''
    events = events.astype(float)
    start = [0, 0, ]
    end = [0, ]
    if not len(start) == len(end):
        dif = len(start) - len(end)
        start = where(events[:, 2] == 150)[0]
        end = where(events[:, 2] == 151)[0]

        # Aborted block during a trial, find location where [start ... start
        # end] occurs
        i_start, i_end = 0, 0   # i_start points to the beginning of the current
        # trial and i_end to the beginning of the current trial.

        if not (len(start) == len(end)):
            # Handle this condition by looking for the closest start to each
            # end.
            id_keep = (0 * events[:, 0]).astype(bool)
            start_times = events[start, 0]
            end_times = events[end, 0]

            for i, e in enumerate(end_times):
                d = start_times - e
                d[d > 0] = -inf
                matching_start = argmax(d)
                evstart = start[matching_start]

                if (151 in events[evstart - 10:evstart, 2]):
                    prev_end = 10 - \
                        where(events[evstart - 10:evstart, 2] == 151)[0][0]
                    id_keep[(start[matching_start] - prev_end + 1):end[i] + 1] = True
                else:
                    id_keep[(start[matching_start] - 10):end[i] + 1] = True
            events = events[id_keep, :]

        start = where(events[:, 2] == 150)[0]
        end = where(events[:, 2] == 151)[0]

    trials = []
    blocks = []
    block = -1
    for i, (ts, te) in enumerate(zip(start, end)):
        # Get events just before trial onset, they mark trial numbers
        trial_nums = events[ts - 8:ts + 1, 2]
        pins = trial_nums[(0 <= trial_nums) & (trial_nums <= 8)]
        if len(pins) == 0:
            trial = 1
        else:
            # Convert pins to numbers
            trial = sum([2**(8 - pin) for pin in pins])
        if trial == 1:
            block += 1
        trials.append(trial)
        blocks.append(block)
    # If the recording did not start before the first trial we might miss trials
    # In this case the first trial should not be labelled 1.
    for b in unique(blocks):
        if sum(blocks == b) is not 100:
            ids = where(blocks == b)[0]
            trials[ids[0]] = trials[ids[1]] - 1
    return events[start, 0], events[end, 0], np.array(trials), np.array(blocks)

val2field = {
    41: 'meg_side', 40: 'meg_side',
    31: 'meg_noise_sigma', 32: 'meg_noise_sigma', 33: 'meg_noise_sigma',
    64: '_onset', 49: 'ref_offset', 50: 'cc', 48: 'stim_offset',
    24: 'button', 23: 'button', 22: 'button', 21: 'button', 88: 'button',
    10: 'meg_feedback', 11: 'meg_feedback'
}


def fname2session(filename):
    print(filename)
    #'/Volumes/dump/conf_data/raw/s04-04_Confidence_20151217_02.ds'
    return int(filename.split('/')[-1].split('_')[-2])


def get_meta(events, tstart, tend, tnum, bnum, day, subject, buttons=None):
    trls = []
    for ts, te, trialnum, block in zip(tstart, tend, tnum, bnum):
        trig_idx = (ts < events[:, 0]) & (events[:, 0] < te)
        trigs = events[trig_idx, :]
        trial = {}
        stim_state = ['stim', 'ref']
        cc_state = list(range(10))[::-1]
        for i, (v, t) in enumerate(zip(trigs[:, 2], trigs[:, 0])):
            if not v in list(val2field.keys()):
                continue
            fname = val2field[v]
            if v == 64:
                fname = stim_state.pop() + fname
            if v == 50:
                fname = fname + str(cc_state.pop())
            trial[fname] = v
            trial[fname + '_t'] = t

        trial['trial'] = trialnum - 1
        trial['block_num'] = block
        trial['start'] = ts
        trial['end'] = te
        trial['day'] = day
        trial['snum'] = subject
        if buttons is not None:
            but_idx = (ts < buttons[:, 0]) & (buttons[:, 0] < te)
            if sum(but_idx) > 0:
                buts = buttons[but_idx, :]
                trial['megbuttons'] = buts[0, 2]
            else:
                trial['megbuttons'] = np.nan
        trls.append(trial)

    trls = pd.DataFrame(trls)

    return trls


def correct_recording_errors(df):
    '''
    Cleanup some of the messy things that occured.
    '''

    if 3 in unique(df.snum):
        id_button_21 = ((df.snum == 3) & (df.button == 21) &
                        ((df.day == 20151207) | (df.day == 20151208)))
        id_button_22 = ((df.snum == 3) & (df.button == 22) &
                        ((df.day == 20151207) | (df.day == 20151208)))
        df.loc[id_button_21, 'button'] = 22
        df.loc[id_button_22, 'button'] = 21
    return df


def cleanup(meta):
    '''
    A meta dataframe contains many duplicate columns. This function removes these
    and checks that no information is lost.
    '''
    cols = []
    # Check button + conf + response
    no_lates = meta.loc[~isnan(meta.button)]
    no_lates = no_lates.query('~(button==88)')

    # Check for proper button to response mappings!
    assert all(no_lates.button.replace(
        {21: 1, 22: 1, 23: -1, 24: -1}) == no_lates.response)
    assert all(no_lates.button.replace(
        {21: 2, 22: 1, 23: 1, 24: 2}) == no_lates.confidence)
    cols += ['button']
    assert all((no_lates.meg_feedback - 10) == no_lates.correct)
    cols += ['meg_feedback']
    assert all((no_lates.meg_side.replace({40: -1, 41: 1})) == no_lates.side)
    cols += ['meg_side']
    assert all((no_lates.meg_noise_sigma.replace(
        {31: .05, 32: .1, 33: .15})) == no_lates.noise_sigma)

    # Also check if meg button is related to button resp.
    map_a = (no_lates.megbuttons.replace(
        {232: 21, 228: 22, 226: 23, 225: 24}).values == no_lates.button.values)
    map_b = (no_lates.megbuttons.replace(
        {232: 24, 228: 23, 226: 22, 225: 21}).values == no_lates.button.values)
    if not (all(map_a) or all(map_b)):
        da = sum(~map_a)
        db = sum(~map_b)
        print('MEG Button aligment not perfect: %i, %i' % (da, db))
    cols += ['meg_noise_sigma']
    cols += ['cc%i' % c for c in range(10)]
    cols += ['meg_side_t', ]
    return meta.drop(cols, axis=1)


def mne_events(data, time_field, event_val):
    return vstack([data[time_field], 0 * data[time_field], data[event_val]]).astype(int).T
