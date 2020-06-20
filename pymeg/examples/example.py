import meg
import mne
import numpy as np


filename = '/Volumes/dump/S4-5_Attractor_20161110_02.ds'
raw = mne.io.read_raw_ctf(filename)

other_pins = {100:'session_number',
        101:'block_start'}

trial_pins = {150:'trial_num'}

mapping = {
       ('noise', 0):111,
       ('noise', 1):112,
       ('noise', 2):113,
       ('start_button', 0):89,
       ('start_button', 1):91,
       ('trial_start', 0):150,
       ('trial_end', 0):151,
       ('wait_fix', 0):30,
       ('baseline_start',0):40,
       ('dot_onset',0):50,
       ('decision_start',0) : 60,
       ('response',-1) : 61,
       ('response',1) : 62,
       ('no_decisions',0) : 68,
       ('feedback',0) : 70,
       ('rest_delay',0) : 80}


mapping = dict((v,k) for k,v in mapping.iteritems())


def get_meta(raw, mapping):
    meta, timing = meg.preprocessing.get_meta(raw, mapping, trial_pins, 150, 151, other_pins)
    blocks = list(where(meta.trial_num.values==1)[0]) +  [len(meta)]
    meta.loc[:, 'hash'] = timing.decision_start_time.values
    return meta, timing

meta, timing = get_meta(raw, mapping)

index = meta.block_start

for i, ((bnum, mb), (_, tb)) in enumerate(zip(meta.groupby(index), timing.groupby(index))):
    # Iterate through blocks in meta and timing
    r = raw.copy()
    r.crop(tmin=(tb.baseline_start_time.min()/1200.)-10, tmax=10+(tb.feedback_time.max()/1200.))
    mb, tb = get_meta(r, mapping)
    r, ants, artdef = meg.preprocessing.preprocess_block(r, blinks=False)

    slmeta, stimlock = meg.preprocessing.get_epoch(r, mb, tb,
        event='decision_start_time',
        epoch_time=(-2.5, .5),
        base_event='decision_start_time',
        base_time=(-3, -2.5),
        epoch_label='decision_start_time')




    del slmeta
    del stimlock
    rlmeta, resplock = meg.preprocessing.get_epoch(r, mb, tb, event='response_time', epoch_time=(-2.5, .5),
        base_event='baseline_start_time', base_time=(0, 0.5), epoch_label='hash')
    rlmeta.to_hdf('resp_meta_%i_ds1.hdf'%i, 'meta')
    resplock.save('resp_meta_%i_ds1-epo.fif.gz'%i)
