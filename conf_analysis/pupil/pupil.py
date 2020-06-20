# -*- coding: utf-8 -*-
'''
Load EDFs and prepare data frames.
'''
import pandas as pd
from scipy import signal
import numpy as np
#from pylab import *
import sympy
import patsy
import seaborn as sns
sns.set_style('ticks')

from . import patsy_transforms as pt
from scipy import interpolate as ipt

EDF_HZ = 1000.0


decision2condition = {21:'1st High', 22:'1st Low', 23:'2nd Low', 24:'2nd High'}

def cleanup(events):
    '''
    Cleans pupil data from blink artifacts and some other things. After cleaning
    data is converted to zscore.
    '''
    pnorm = (events.pa - events.pa.mean())/events.pa.std()
    vpa = np.concatenate([[0], np.diff(pnorm)])
    errors = (events.blink.values > 0.5) | (abs(vpa) > 0.5)
    filtered = interp_blinks(events, errors, 5, 5, 2, 'pa')
    events['pac'] = filtered
    #events['pac'] = (filtered - events.pa.mean())/events.pa.std()
    return events

def interp_blinks(events, err_source, pre, post, offset=10, field='pa'):
    '''
    Linearly interpolate blinks
    '''
    d = np.diff(err_source)
    blinks = np.where(d)[0].tolist()
    if err_source[0] == 1:
        #Starts with a blink that is missing in blinks. Add it
        blinks.insert(0, 0)
    if err_source[-1] == 1:
        #Ends with blink that will not be detected. Add it
        blinks.append(len(err_source))

    if not (np.mod(len(blinks), 2)==0):
        raise RuntimeError('Number of errs not even')
    filtered = events[field].copy()
    for start, end in zip(blinks[0::2], blinks[1::2]):
        filtered.values[start-pre:end+post] = np.nan

    idx = np.arange(len(filtered))
    idnan = np.isnan(filtered.values)
    filtered = ipt.splev(idx, ipt.splrep(idx[~idnan], filtered.values[~idnan], k=1) )
    return filtered

def make_design_matrix(events):
    '''
    Make a design matrix for regression.
    '''
    pass


def decimate(data, factor, **kwargs):
    '''
    Donwsample a data frame by downsampling all columns.
    Forces the use of FIR filter to avoid a phase shift.
    Removes the firs factor samples to avoid edge artifacts in the beginning.
    '''
    target = {}
    kwargs['ftype'] = 'fir'
    for column in data.columns:
        target[column] = signal.decimate(data[column], factor, **kwargs)[factor:]
    index = signal.decimate(data.index.values, factor, **kwargs)[factor:]
    return pd.DataFrame(target, index=index)


def filter_pupil(pupil, sampling_rate, highcut = 10., lowcut = 0.1, order=3):
    """
    Band pass filter using a butterworth filter of order 3.

    lowcut: Cut off everything slower than this
    highcut: Cut off everything that is faster than this

    Returns:
        below: everything below the passband
        filtered: everything in the passband
        above: everything above the passband

    Based on scipy cookbook: https://scipy.github.io/old-wiki/pages/Cookbook/ButterworthBandpass
    """

    def butter_bandpass(lowcut, highcut, fs, order=5):
       nyq = 0.5 * fs
       low = lowcut / nyq
       high = highcut / nyq
       b, a = signal.butter(order, [low, high], btype='band')
       return b, a

    b, a = butter_bandpass(lowcut, highcut, sampling_rate, order)
    filt_pupil = signal.filtfilt(b, a, pupil)

    b, a = butter_bandpass(highcut, 0.5*sampling_rate-0.5, sampling_rate, order)
    above = signal.filtfilt(b, a, pupil)
    return pupil - (filt_pupil+above), filt_pupil, above

def eval_model(model, data, summary=True):
    from sklearn import linear_model
    import statsmodels.api as sm

    if len(np.unique(data.left_gx[~np.isnan(data.left_gx)])) == 1:
        model = model.replace('left', 'right')
    y,X = patsy.dmatrices(model, data=data.copy(), eval_env=1)

    m = linear_model.LinearRegression()
    idnan = np.isnan(y.ravel())
    mod = sm.OLS(y[~idnan, :], X[~idnan, :])
    res = mod.fit()
    #print res.summary(xname=X.design_info.column_names)
    m.fit(X[~idnan,:],y[~idnan,:])
    yh = m.predict(X)
    if summary:
        print('R**2:', np.corrcoef(y.ravel(), yh.ravel())[0,1]**2)
    return m, yh, y, X, res



def IRF_pupil(fs=100, dur=4, s=1.0/(10**26), n=10.1, tmax=.930):
    """
    Canocial pupil impulse fucntion [/from JW]

    dur: length in s

    """

    # parameters:
    timepoints = np.linspace(0, dur, dur*fs)

    # sympy variable:
    t = sympy.Symbol('t')

    # function:
    y = ( (s) * (t**n) * (np.math.e**((-n*t)/tmax)) )

    # derivative:
    y_dt = y.diff(t)

    # lambdify:
    y = sympy.lambdify(t, y, "numpy")
    y_dt = sympy.lambdify(t, y_dt, "numpy")

    # evaluate and normalize:
    y = y(timepoints)
    y = y/np.std(y)
    y_dt = y_dt(timepoints)
    y_dt = y_dt/np.std(y_dt)

    # dispersion:
    y_dn = ( (s) * (timepoints**(n-0.01)) * (np.math.e**((-(n-0.01)*timepoints)/tmax)) )
    y_dn = y_dn / np.std(y_dn)
    y_dn = y - y_dn
    y_dn = y_dn / np.std(y_dn)

    return y, y_dt, y_dn


def prepare_glm_regressors(events, messages):
    events.sortlevel(level='subject', inplace=True, axis=1)
    messages.sortlevel(level='subject', inplace=True, axis=1)
    print(messages.index.names)
    print(events.index.names)

    def make_index(field):
        md = field.reset_index()
        md = md.rename(columns={field.name:'sample_time'})
        md.sample_time = md.sample_time.astype(int)
        md = md.set_index(['session', 'block', 'subject', 'sample_time'])
        return md
    # Add reference contrast event
    md = make_index(messages.decision_start_time)
    events['ref'] = pt.event_ramp().transform(events.decision, start=md, end=md, pre=1900, post=-1500)

    # Add events for a decision ramp. At the moment it is a boxcar.
    mde = make_index(messages.decision_time)

    events['decramp'] =  pt.event_ramp().transform(events.decision, start=md, end=mde, pre=0, post=0, ramp='downramp')
    events['decramp2'] =  pt.event_ramp().transform(events.decision, start=md, end=mde, pre=0, post=0, ramp='boxcar')
    events['dec_start'] =  pt.event_ramp().transform(events.decision, start=md, end=md, pre=0, post=0, ramp='boxcar')

    # Add events for a individual decision ramps. At the moment it is a boxcar.
    # Add events for individual decisions
    for d in [21,22,23,24]:
        md = make_index(messages[messages.decision==d].decision_start_time)
        mde = make_index(messages[messages.decision==d].decision_time)
        events['decramp%i'%d] =  pt.event_ramp().transform(events.decision, start=md, end=mde, pre=0, post=0, ramp='boxcar')
        events['decstart%i'%d] =  pt.event_ramp().transform(events.decision, start=md, end=md, pre=0, post=0, ramp='boxcar')
        events['decision%i'%d] =  pt.event_ramp().transform(events.decision, start=mde, end=mde, pre=0, post=0, ramp='boxcar')

    # Add events for feedback

    fde = make_index(messages.feedback_time[(~np.isnan(messages.feedback_time)) & (messages.feedback.values==-1)])
    events['feedback_offset_neg'] =  pt.event_ramp().transform(events.feedback, start=fde, end=fde,
                                                          pre=0, post=0, ramp='boxcar')
    fde = make_index(messages.feedback_time[(~np.isnan(messages.feedback_time)) & (messages.feedback.values==1)])
    events['feedback_offset_pos'] =  pt.event_ramp().transform(events.feedback, start=fde, end=fde,
                                                          pre=0, post=0, ramp='boxcar')
    events.feedback_offset_neg.ix[np.isnan(events['feedback_offset_neg'])] = 0
    events.feedback_offset_pos.ix[np.isnan(events['feedback_offset_pos'])] = 0
    return events, messages


def fasta2(events, messages, field='decision_time', pre=1., post=1, Hz=100):
    # Only create the necessary index in a separate dataframe.
    number = np.nan*np.ones((len(events),))
    time = np.nan*np.ones((len(events),))
    time_t = np.linspace(-pre, post, pre*Hz + post*Hz)
    positions = get_locations(events, messages, field)
    for i, p in enumerate(positions):
        start = p-pre*Hz
        end = p+post*Hz
        number[start:end] = i
        if end > len(time):
                time[start:end] = time_t[:len(time[start:end])]
        else:
            time[start:end] = time_t
    return number, time

def expand_field(events, messages, values, start, end):
    # Expand a field in messages
    field = np.nan*np.ones((len(events),))
    start_positions = get_locations(events, messages, start)
    end_positions = get_locations(events, messages, end)
    assert len(start_positions) == len(end_positions) == len(values)
    print(len(start_positions), len(end_positions), len(values))
    for i, (s, e, v) in enumerate(zip(start_positions, end_positions, values)):
        if not s<e:
            print(i, s, e, v)
            return None
        field[s:e] = v
    print(s, e, v)
    return field

def find_closest_index(sample_times, idx, offset):
    sample_times = sample_times.values[offset:]
    pos = np.argmin(abs(sample_times-idx[-1]))
    return offset+pos


def get_locations(events, messages, field='decision_time'):
    '''
    Get the indices in events for time points specified in messages.

    Returns a list with indices
    '''
    idx = pd.IndexSlice
    offset = 0
    pos = []
    for (session, block, subject), ev in events.groupby(level=['session', 'block', 'subject']):
        index = [(session, block, subject, int(time))
                            for (session, block, subject, _), time in
                                    messages.loc[idx[session, block, subject, :], field].items()]
        st = ev.index.get_level_values('sample_time')
        pos.extend([offset + find_closest_index(st, i, 0) for i in index])
        offset += len(ev)
    return pos
