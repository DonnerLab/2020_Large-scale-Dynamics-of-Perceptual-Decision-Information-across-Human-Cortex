import patsy
import numpy as np

class empty_transform(object):
    '''
    Transforms events into
    '''
    def __init__(self):
        pass

    def memorize_chunk(self, *args, **kwargs):
        pass

    def memorize_finish(self, *args, **kwargs):
        pass


class Zscore(empty_transform):
    def transform(self, x):
        return (x-np.nanmean(x))/np.nanstd(x)


class DT(empty_transform):
    def transform(self, x):
        return np.r_[[0], np.diff(x)]


class Cc_t(empty_transform):
    def transform(self, x, levels=None):
        if levels is None:
            levels = np.unique(x)
        out = np.zeros((len(x), len(levels)))
        for i, level in enumerate(levels):
            idx = x==level
            out[:, i] = idx

        return out


class boxcar_t(empty_transform):
    def transform(self, x, pre=10, post=10, val='normalized'):
        if val == 'normalized':
            val = 1./(pre+post)
        idx = np.where(x)[0]
        for index in idx:
            x[idx-pre:idx+post] = val
        return x


class ramp_t(empty_transform):
    def transform(self, x, pre=10, post=10, ramp_type='upramp', start=0., end=1.):
        try:
            x = x.values
        except AttributeError:
            pass
        vals = start + np.arange(pre+post) * (end/(pre+post))
        if ramp_type == 'upramp':
            pass
        elif ramp_type == 'downramp':
            vals = vals[::-1]
        else:
            vals = ramp_type(np.arange(pre+post))
        if len(x.shape) == 1:
            idx = np.where(x[:])[0].ravel()
            for index in idx:
                x[index-pre:index+post] = vals
        else:
            for i in range(x.shape[1]):
                idx = np.where(x[:,i])[0].ravel()
                for index in idx:
                    x[index-pre:index+post, i] = vals
        return x


class event_ramp(empty_transform):
        def transform(self, x, start, end, pre=10, post=10, ramp='boxcar'):
            # This is a bit tricky because start and end can be different things
            # For now I assume that start and end are indices (maybe multi-indices)
            # Pre and post are in sample space
            x = x*0
            for s, e in zip(start.index.values, end.index.values):
                s = tuple(s[:-1] + ((s[-1]-pre),))
                e = tuple(e[:-1] + ((e[-1]+post),))
                v=1
                if ramp=='upramp':
                    v = np.linspace(0,1,len(x.loc[s:e]))
                elif ramp=='downramp':
                    v = np.linspace(1,0,len(x.loc[s:e]))
                x.loc[s:e] = v
            return x


class convolution(empty_transform):
    '''
    A stateful transform for patsy that convolves regressors with some function

    TODO: Documentation.
    '''
    def __init__(self):
        pass

    def transform(self, x, func=[1]):
        try:
            x = x.values
        except AttributeError:
            pass
        func = np.pad(func, [len(func), 0], mode='constant')
        if len(x.shape) > 1:
            out = np.array([np.convolve(t, func, mode='same') for t in x.T])
            return out.T
        return np.convolve(x, func, mode='same')


class multi_convolution(empty_transform):
    '''
    A stateful transform for patsy that convolves regressors with some function

    TODO: Documentation.
    '''
    def __init__(self):
        pass

    def transform(self, x, func=[1]):
        try:
            x = x.values
        except AttributeError:
            pass
        func = np.asarray(func)
        if len(func.shape)==1:
            out = self.conv(x, func)
            return out.T
        else:
            out = np.vstack([self.conv(x, f) for f in func])
            return out.T

    def conv(self, x, f):
        func = np.pad(f, [len(f), 0], mode='constant')
        if len(x.shape) > 1:
            out = np.vstack([np.convolve(t, f, mode='same') for t in x.T])
            return out
        out = np.convolve(x, func, mode='same')
        return out


class spline_convolution(object):
    '''
    A stateful transform for patsy that convolves regressors with a spline basis
    function.

    TODO: Documentation.
    '''
    def __init__(self):
        pass

    def memorize_chunk(self, x, **kwargs):
        pass

    def memorize_finish(self, **kwargs):
        pass

    def transform(self, x, degree=2, df=5, length=100):
        try:
            x = x.values
        except AttributeError:
            pass
        knots = [0, 0] + np.linspace(0,1,df-1).tolist() + [1,1]
        basis = patsy.splines._eval_bspline_basis(np.linspace(0,1,length), knots, degree)
        basis /= basis.sum(0)
        if len(x.shape) > 1 and not (x.shape[1] == 1):
            return np.r_[[self.conv_base(t, basis, length) for t in x]]
        else:
            return self.conv_base(x.ravel(), basis, length)

    def conv_base(self, x, basis, length):
        out = np.empty((len(x), basis.shape[1]))
        for i, base in enumerate(basis.T):
            out[:,i] = np.convolve(x, np.pad(base, [length, 0], mode='constant'), mode='same')
        return out


class spline_event(object):
    '''
    Puts a spline basis function at specified timepoints into a time series
    '''
    def __init__(self):
        pass

    def memorize_chunk(self, x, **kwargs):
        pass

    def memorize_finish(self, **kwargs):
        pass

    def transform(self, x, events=[0], degree=2, df=5, length=100, values=None):
        '''
        X contains time points of the time series.
        Events is an array if time points where spline basis functions should
        be placed.
        Length is the duration of each basis function. Can be an array to specify
        individual durations. This will make interpretation of parameters difficult
        though.
        '''
        if values is None:
            values = events*0+1
        knots = [0, 0] + np.linspace(0,1,df-1).tolist() + [1,1]
        if type(length)==int:
            basis = patsy.splines._eval_bspline_basis(np.linspace(0,1,length), knots, degree)
            basis /= basis.sum(0)
            duration = length
        colum = np.zeros((x.shape[0], df))
        for i, event in enumerate(events):
            if not type(length)==int:
                basis = patsy.splines._eval_bspline_basis(np.linspace(0,1,length[i]), knots, degree)
                basis /= basis.sum(0)
                duration = length[i]

            idx = np.argmin(abs(x-event))
            tend = min(idx+duration, len(x))
            bend = min(basis.shape[0], len(x)-idx)
            colum[idx:tend, :] += basis[:bend, :]*values[i]

        return colum


class flat_event(object):
    '''
    Puts a boxcar at specified timepoints into a time series
    '''
    def __init__(self):
        pass

    def memorize_chunk(self, x, **kwargs):
        pass

    def memorize_finish(self, **kwargs):
        pass

    def transform(self, x, events=[0], length=100, values=None):
        '''
        X contains time points of the time series.
        Events is an array if time points where spline basis functions should
        be placed.
        Length is the duration of each basis function. Can be an array to specify
        individual durations. This will make interpretation of parameters difficult
        though.
        '''
        if values is None:
            values = events*0+1
        if type(length)==int:
            duration = length
        colum = np.zeros((x.shape[0],))
        for i, event in enumerate(events):
            if not type(length)==int:
                duration = length[i]
            idx = np.argmin(abs(x-event))
            tend = min(idx+duration, len(x))
            colum[idx:tend] = values[i]

        return colum


class contrast_onsets(object):
    def __init__(self):
        pass

    def memorize_chunk(self, x, **kwargs):
        pass

    def memorize_finish(self, **kwargs):
        pass

    def transform(self, x, dur=0.2, vals=None, default=0):
        try:
            x = x.values
        except AttributeError:
            pass
        if vals is None:
            vals=[1]*10
        xb = []
        for k, v in zip(np.arange(0., 1, 0.1), vals):
            if dur<0:
                idx = x*0+default
                idx[np.argmin(abs(x-k))]=v
            else:
                index = ((k<= x) & (x<k+dur))
                idx = x*0+default
                idx[index] = v

            xb.append(idx.astype(float))

        return np.vstack(xb).T

Cc = patsy.stateful_transform(Cc_t)
box = patsy.stateful_transform(boxcar_t)
ramp = patsy.stateful_transform(ramp_t)
BS = patsy.stateful_transform(spline_convolution)
SE = patsy.stateful_transform(spline_event)
FE = patsy.stateful_transform(flat_event)
CO = patsy.stateful_transform(contrast_onsets)

F = patsy.stateful_transform(convolution)
MF = patsy.stateful_transform(multi_convolution)
Z = patsy.stateful_transform(Zscore)
evramp = patsy.stateful_transform(event_ramp)
dt = patsy.stateful_transform(DT)
