'''
Implement tests for models that use a non-linear transformation of contrast

'''
import matplotlib
import sys
sys.path.append('/Users/nwilming/u')

import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns

from conf_analysis.behavior import empirical
from scipy.optimize import minimize
from scipy.special import erf
from scipy.special import erfinv
from scipy.special import expit
from scipy.special import logit
from scipy.stats import norm

try:
    import pymc3 as pm
    import theano.tensor as T
except ImportError:
    pm = None
    T = None


from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy.stats import bernoulli


sns.set_style('ticks')


def get_data():
    data = empirical.load_data()
    data = empirical.data_cleanup(data)
    data['conf_0index'] = data.confidence - 1
    idnan = (np.isnan(data.conf_0index.values) |
             np.isnan(data.response))
    data = data.loc[~idnan, :]
    data.loc[:, 'absmc'] = np.abs(data.mc)
    return data


def power_transformer(x, a):
    '''
    x is in range (0, 1), maps to (0, 1)
    a is in range(-inf, inf)
    '''
    y = x**np.exp(a)
    return y


transformer = power_transformer


def bias(x, slope, intercept):
    '''
    Shift and scale contrast such that they go into [0, 1].
    x is in [0, 1], mean = 0.5
    '''
    x = 2 * (x - 0.5)  # [-.5, .5]
    x = x * slope + intercept  # [-inf, inf]
    y = (expit(x) + 1.)
    return y


def base_model(offset, a, *tslope, **kwargs):
    contrast = kwargs['contrast']
    # Transform contrast and remove ref such that 0 is ideal decision boundary
    transformed = transformer(contrast, a) - transformer(0.5, a)
    weights = np.array(tslope)
    transformed = weights[np.newaxis, :] * transformed
    y = transformed.mean(1) + offset
    return y


def variance_model(offset, varslope, a, *tslope, **kwargs):
    contrast = kwargs['contrast']
    # Transform contrast and remove ref such that 0 is ideal decision boundary
    transformed = transformer(contrast, a) - transformer(0.5, a)
    weights = np.array(tslope).astype(float)
    transformed = weights[np.newaxis, :] * transformed
    # factor #(weights.mean())#factor
    stdc = (transformed.std(1)) / transformed.std(1).mean()
    y = transformed.mean(1) + weights.mean() * varslope * stdc + offset
    return y


def bayes_variance_model(data, var=True):
    '''
    Implement a model that predicts individual choices based on
    non-linearly transformed contrast samples.

    If var == True the variance of the ten contrast samples is
    explicitly modeled. If False it is ommited.
    '''
    basic_model = pm.Model()
    contrast = get_contrast(data)
    response = (data.response + 1) / 2.
    with basic_model:
        # Priors for unknown model parameters
        beta_time = pm.Normal('time', mu=0, sd=10, shape=10)
        offset = pm.Normal('offset', mu=0, sd=0.05)
        alpha = pm.Normal('alpha', mu=0, sd=2)

        transformed = contrast**pm.math.exp(alpha) - 0.5**pm.math.exp(alpha)
        weighted = transformed * beta_time[np.newaxis, :]
        # Expected value of outcome
        mu = weighted.mean(1) + offset
        if var:
            beta_std = pm.Normal('std', mu=0, sd=.35)
            stdc = (transformed).std(1) / transformed.std(1).mean()
            mu += beta_std * beta_time.mean() * stdc

        ps = pm.math.invlogit(mu)
        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Bernoulli('Y_obs', ps, observed=response)
    return basic_model


def get_contrast(data):
    '''
    Stack contrast values in a data frame and take care of
    values that are outside of [0, 1]
    '''
    contrast = np.vstack(data.contrast_probe)
    contrast[contrast < 0] = 0
    contrast[contrast > 1] = 1
    return contrast


def get_surfaces(x, y, filter, bins):
    '''
    Compute how often x and y occured before a specific
    choice.

    Parameters
    ----------
    x, y: array containing mean contrast and std respectively
    filter: array
        True for choice 1, false for choice 2
    bins: 2-list of arrays
        Specifies edges for the 2D histogram. 
    '''
    idx = filter > 0
    resp1 = plt.histogram2d(x[idx], y[idx],
                            bins=bins)[0]
    resp2 = plt.histogram2d(x[~idx], y[~idx],
                            bins=bins)[0]
    return resp1, resp2


def get_predicted_surface(x, y, predicted, bins, mincnt=10):
    '''
    
    '''
    cnt = np.histogram2d(x, y, bins=bins)[0]
    cnt[cnt < mincnt] = np.nan
    return (np.histogram2d(x, y, weights=predicted, bins=bins)[0] / cnt)


def tovarparams(df):
    offset, varslope, a = df.loc[:, 'offset'], df.loc[
        :, 'std'], df.loc[:, 'alpha']
    time = np.vstack(df.loc[:, 'time'])
    return np.concatenate((offset[:, np.newaxis],
                           varslope[:, np.newaxis],
                           a[:, np.newaxis], time), axis=1)


def tobaseparams(df):
    offset, a = df.loc[:, 'offset'], df.loc[:, 'alpha']
    time = np.vstack(df.loc[:, 'time'])
    return np.concatenate((offset[:, np.newaxis],
                           a[:, np.newaxis], time), axis=1)


def get_all_subs(d, models=[True, False], iter=5000):
    '''
    Estimate the bayesian model for all subjects and save results.
    '''
    import pickle
    res = []
    for s, kd in d.groupby('snum'):
        for var in models:
            with bayes_variance_model(kd, var=var) as model:
                trace = pm.sample(iter, init='advi',
                                  njobs=3)
                dic = pm.stats.dic(trace)
                waic = pm.stats.waic(trace)
                bpic = pm.stats.bpic(trace)
                GR = pm.diagnostics.gelman_rubin(trace)

                fname = 's%i_var%s_modeltrace.pickle' % (s, var)
                pickle.dump({'trace': trace, 'model': model, 'dic': dic,
                              'waic': waic.WAIC, 'waic_se': waic.WAIC_se,
                              'p_waic': waic.p_WAIC, 'bpic': bpic,
                              'gelman_rubin': GR},
                             open(fname, 'w'))
            dd = {'snum': s, 'var': var}

            for var in trace.varnames:
                dd[var] = trace[var]
            dd['time'] = [k for k in dd['time']]
            dd = pd.DataFrame(dd)
            dd.to_hdf('bayes_snum%i_var%s_fits.hdf' % (s, str(var)), 'df')
            res.append(pd.DataFrame(dd))
    return res


'''
Here start the plotting functions
'''


def plot_model(df, model=None, bins=[np.linspace(0, .3, 26), np.linspace(0, 1, 26)],
               hyperplane_only=False, alpha=1, mincnt=10, cmap='RdBu_r',
               fieldy='stdc', fieldx='mc', vmin=-2.4, vmax=2.4,
               predicted_surface=False, contrast=None):
    C, M = np.meshgrid(*bins)
    if model is not None:
        if not predicted_surface:
            N = 50
            resp1 = np.ones((len(bins[0]) - 1, len(bins[1]) - 1))
            resp2 = np.ones((len(bins[0]) - 1, len(bins[1]) - 1))
            if contrast is None:
                contrast = get_contrast(df)
            ps = expit(model(contrast))
            for i in range(N):
                idx = bernoulli.rvs(ps).astype(bool)

                resp1 += plt.histogram2d(df[fieldy].values[idx],
                                         df[fieldx].values[idx],
                                         bins=bins)[0]
                resp2 += plt.histogram2d(df[fieldy].values[~idx],
                                         df[fieldx].values[~idx],
                                         bins=bins)[0]
            resp2[(resp2) <= (mincnt * N)] = np.nan
            resp1[(resp1) <= (mincnt * N)] = np.nan
            resp1, resp2 = resp1 / N, resp2 / N
            resp1 = resp1.astype(float) / np.nansum(resp1)
            resp2 = resp2.astype(float) / np.nansum(resp2)

            plane = np.log(resp1 / resp2)
            plane[plane == 1] = np.nan
        else:
            if contrast is None:
                contrast = get_contrast(df)
            ps = expit(model(contrast))
            plane = get_predicted_surface(contrast.std(1), contrast.mean(1),
                                          ps, bins, mincnt)

    else:
        idx = df.response > 0
        resp1 = plt.histogram2d(df[fieldy].values[idx],
                                df[fieldx].values[idx],
                                bins=bins)[0] + 1
        resp1[resp1 == 1] = np.nan
        resp2 = plt.histogram2d(df[fieldy].values[~idx],
                                df[fieldx].values[~idx],
                                bins=bins)[0] + 1
        resp2[resp2 == 1] = np.nan
        resp1[resp1 < mincnt] = np.nan
        resp2[resp2 < mincnt] = np.nan
        resp1 = resp1.astype(float) / np.nansum(resp1)
        resp2 = resp2.astype(float) / np.nansum(resp2)

        plane = np.log(resp1 / resp2)
        plane[plane == 1] = np.nan
    pcol = plt.pcolormesh(bins[1], bins[0],
                          np.ma.masked_invalid(plane), cmap=cmap,
                          vmin=vmin, vmax=vmax, rasterized=True, linewidth=0)
    pcol.set_edgecolor('face')
    if predicted_surface:
        pcol = plt.contour(bins[1][1:], bins[0][1:],
                           np.ma.masked_invalid(plane), [0.35, 0.5, 0.65])

    mind, maxd = plt.xlim()
    plt.ylim(bins[0][0], bins[0][-1])
    plt.xlim(bins[1][0], bins[1][-1])
    plt.plot([0, 0], [bins[0][0], bins[0][-1]], 'k--', lw=2)


def plot_bayes_model(df, params, model='variance',
                     bins=[np.linspace(0, .3, 51), np.linspace(0, 1, 51)],
                     cmap='RdBu_r', stride=20, start=0, vmin=-2.4, vmax=2.4):
    if model == 'variance':
        func = variance_model
        params = tovarparams(params.query('var==True'))[start::stride, :]

    elif model == 'base':
        func = base_model
        params = tobaseparams(params.query('var==False'))[start::stride, :]

    else:
        raise RuntimeError("don't know the model")
    contrast = get_contrast(df)
    resp1 = 0 * np.ones((len(bins[0]) - 1, len(bins[1]) - 1)).astype(float)
    resp2 = 0 * np.ones((len(bins[0]) - 1, len(bins[1]) - 1)).astype(float)

    print('Using %i models' % (len(params)))
    for row in params:
        ps = expit(func(*row, contrast=contrast))
        response = bernoulli.rvs(ps).astype(bool)
        r1, r2 = get_surfaces(df.stdc.values, df.mc.values, response, bins)
        resp1, resp2 = resp1 + r1, resp2 + r2

    resp1 = resp1.astype(float) / np.nansum(resp1)
    resp2 = resp2.astype(float) / np.nansum(resp2)
    plane = np.log(resp1 / resp2)

    plane[plane == 0] = np.nan

    pcol = plt.pcolormesh(bins[1], bins[0], np.ma.masked_invalid(plane), cmap=cmap,
                          vmin=vmin, vmax=vmax, rasterized=True, linewidth=0)
    pcol.set_edgecolor('face')

    mind, maxd = plt.xlim()
    plt.ylim(bins[0][0], bins[0][-1])
    plt.xlim(bins[1][0], bins[1][-1])
    plt.plot([0, 0], [bins[0][0], bins[0][-1]], 'k--', lw=2)


def plot_subject(d, df, start=1000, row=None, gs=None):
    print(np.unique(df['var']))
    df = df.iloc[start:, :]
    print(np.unique(df['var']))
    if row is None or gs is None:
        import matplotlib
        gs = matplotlib.gridspec.GridSpec(1, 6)
        row = 0
    low, high = np.percentile(d.mc, [5, 95])
    slow, shigh = np.percentile(d.stdc, [5, 95])
    bins = [np.linspace(slow, shigh, 26), np.linspace(low, high, 26)]
    plt.subplot(gs[row, 0])
    # try:
    plot_bayes_model(d, df, 'base', bins=bins)
    # except ValueError:
    #    pass
    plt.xticks(np.around([low, .5, high], 2))
    plt.yticks(np.around([slow, shigh], 2))
    if row == 0:
        plt.title('base')
    plt.subplot(gs[row, 1])
    plot_bayes_model(d, df, 'variance', bins=bins)

    plt.xticks(np.around([low, .5, high], 2))
    plt.yticks(np.around([slow, shigh], 2))
    if row == 0:
        plt.title('var')
    plt.subplot(gs[row, 2])
    bins = [np.linspace(slow, shigh, 11), np.linspace(low, high, 11)]
    plot_model(d, bins=bins, mincnt=2)
    plt.xticks(np.around([low, .5, high], 2))
    plt.yticks(np.around([slow, shigh], 2))
    if row == 0:
        plt.title('empirical')

    plt.subplot(gs[row, 4:6])
    colors = {0: 'b', 1: 'r'}
    for j, var in enumerate([True, False]):
        try:
            w = np.vstack(df.query('var==%s' % var).time)
            for t, weights in enumerate(w.T):
                assert len(weights) > 10
                plt.plot([1 + t + j / 3., 1 + t + j / 3.], np.percentile(weights,
                                                                         [5, 95]), color=colors[j], dash_capstyle='round', lw=5)
                plt.plot([1 + t + j / 3.], np.percentile(weights,
                                                         [50]), 'o', color=colors[j])
        except ValueError:
            pass
    plt.xlim([.5, 10.5])
    plt.subplot(gs[row, 3])
    sns.kdeplot(df.query('var==True')['alpha'], label='Alpha VAR')
    sns.kdeplot(df['std'])
    try:
        sns.kdeplot(df.query('var==False')['alpha'], label='Alpha BASE')
    except:
        pass
    plt.xticks([-1, 0, 1, 2])
    plt.yticks([])
    sns.despine(left=True)
    plt.legend(bbox_to_anchor=(6.2, 1.))
    plt.plot(np.percentile(df.query('var==True')['alpha'], [
             5, 95]),  [-.25, -.25], 'b', dash_capstyle='round', lw=5)
    plt.plot(np.percentile(df.query('var==True')['alpha'], [
             50]), [-.25], 'bo', label='Alpha Var')
    try:
        plt.plot(np.percentile(df.query('var==False')['alpha'], [
                 5, 95]), [-0.5, -0.5], 'r', dash_capstyle='round', lw=5)
        plt.plot(np.percentile(df.query('var==False')['alpha'], [
                 50]), [-0.5], 'ro', label='Alpha Base')
    except:
        pass
    plt.plot(np.percentile(df.query('var==True')['std'], [
             5, 95]), [-.1, -.1], 'g', dash_capstyle='round', lw=5)
    plt.plot(np.percentile(df.query('var==True')[
             'std'], [50]), [-.1], 'go', label='STD')
    plt.xlim([-0.5, 1.5])
    plt.ylim([-0.75, 7.5])
    plt.yticks([])
    plt.xticks([-0.5, 0, 1.])
    sns.despine(left=True)
    plt.tight_layout()


def model_comparison_plot(resdf, d):
    plt.figure(figsize=(8, 12))
    #gs = matplotlib.gridspec.GridSpec(8, 6)
    cnt = 1
    x = np.linspace(0, 1, 100)
    nmodels = 2
    for i, row in resdf.iterrows():
        snum = int(row.snum)
        plt.subplot(8, 2 * (nmodels + 2), cnt)
        cnt += 1
        kd = d.query('snum==%i' % row.snum)
        contrast = get_contrast(kd)
        mcs = contrast.mean(1)
        low, high = np.percentile(mcs, [5, 95])
        plot_model(kd, lambda x: base_model(
            row.boffset, row.ba, *row.bweights,
            contrast=contrast),
            cmap='RdBu_r', bins=[np.linspace(0, 0.3, 21),
                                 np.linspace(low, high, 21)])

        plt.xticks([0.5])
        plt.yticks([])
        plt.subplot(8, 2 * (nmodels + 2), cnt)
        cnt += 1
        plot_model(kd, lambda x: variance_model(
            row.voffset, row.varslope, row.va, *row.vweights,
            contrast=contrast),
            cmap='RdBu_r', bins=[np.linspace(0, 0.3, 21),
                                 np.linspace(low, high, 21)])
        plt.title('snum=%i' % snum)
        plt.xticks([0.5])
        plt.yticks([])
        plt.subplot(8, 2 * (nmodels + 2), cnt)
        cnt += 1
        plot_model(kd,
                   cmap='RdBu_r', bins=[np.linspace(0, 0.3, 11),
                                        np.linspace(low, high, 11)])
        plt.xticks([0.5])
        plt.yticks([])
        plt.subplot(8, 2 * (nmodels + 2), cnt)
        plt.plot(x, transformer(x, row.ba), label='base')
        plt.plot(x, transformer(x, row.va), label='var')
        #plt.plot(x, transformer(x, row.wa), label='weighted_var')
        plt.xticks([0, 1])
        plt.yticks([0, 1])
        cnt += 1
    plt.legend(bbox_to_anchor=(2.9, 1.))
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(
            '/Users/nwilming/u/conf_analysis/plots/nonlinear_modelfits.pdf')


def parameter_overview(d):
    con = np.vstack([norm.rvs(0.5 + 0.05, v, size=(50000, 10)) for v in [0.05, 0.1, 0.15]] +
                    [norm.rvs(0.5 - 0.05, v, size=(50000, 10)) for v in [0.05, 0.1, 0.15]])
    con[con > 1] = 1
    con[con < 0] = 0
    plt.figure(figsize=(8, 8))
    a = 10
    gs = matplotlib.gridspec.GridSpec(9, 9)

    for j, std in enumerate(np.linspace(-.25, .25, 9)):
        for i, alpha in enumerate(np.linspace(-1.5, 1.5, 9)):
            plt.subplot(gs[i, j])
            if j == 0:
                plt.ylabel('%2.2f' % alpha)
            if i == 0:
                plt.title('%2.2f' % std)
            avals = 1 + 0 * np.linspace(1, 0.5, 10)
            avals = a * avals / avals.mean()
            params = [0, std, alpha] + list(avals)

            plot_model(d, lambda x: variance_model(*params, contrast=x), cmap='RdBu_r',
                       bins=[np.linspace(0, 0.25, 26),
                             np.linspace(.3, .7, 26)],
                       predicted_surface=True, vmin=0, vmax=1, contrast=con)

            plt.xticks([])
            plt.yticks([])

    sns.despine(left=True, bottom=True)
    plt.gcf().text(0.5, 1.015, r'$\beta_{\sigma}$', ha='center', fontsize=20)
    plt.gcf().text(-0.025, 0.5, r'$\alpha$', va='center', rotation='vertical', fontsize=20)
    plt.tight_layout()
    plt.savefig(
        '/Users/nwilming/u/conf_analysis/plots/prameter_overview.pdf', bbox_inches='tight')
    plt.savefig(
        '/Users/nwilming/u/conf_analysis/plots/prameter_overview.svg', bbox_inches='tight')


def rowplot(df, scale=10):
    varcol = sns.xkcd_palette(['blood orange', 'light blue'])
    scale = {'std': 5, 'alpha': 25, 'offset': 25}
    xlims = {'std': [-.05, .1], 'alpha': [-.5, 1.5], 'offset': [-.5, .5]}
    from scipy.stats import gaussian_kde
    for (s, v), d in df.groupby(['snum', 'var']):
        for i, param in enumerate(['std', 'alpha', 'offset']):
            plt.subplot(1, 3, i + 1)
            ys = d[param].values
            low, high = np.percentile(ys, [0.5, 99.5])
            credible = not ((low < 0) and (0 < high))
            if sum(np.isnan(ys)) == len(ys):
                continue
            x = np.linspace(-2, 2, 1000)
            kde = gaussian_kde(d[param].values)
            y = kde(x)

            y0 = y * 0 + d.snum.values[0]
            y1 = y0 + scale[param] * (y / y.sum())
            if (i == 2) and (s == 1):
                labels = {True: 'Variance', False: 'Base'}
                plt.fill_between(x, y0, y1, alpha=0.5,
                                 color=varcol[v], label=labels[v])
            else:
                plt.fill_between(x, y0, y1, alpha=0.5, color=varcol[v])
            if credible:
                plt.plot(x, y1, color='k', lw=1, alpha=0.5)
            plt.title(param)
            plt.xticks([xlims[param][0]] + [0] + [xlims[param][1]])
            plt.xlim(xlims[param])
            if i > 0:
                plt.yticks([])
                sns.despine(left=True, ax=plt.gca())
            if i == 0:
                plt.yticks([5, 10, 15])
                plt.ylabel('Subject')
                sns.despine()
            if i == 1:
                plt.xlabel('Parameter estimate')
    plt.legend(bbox_to_anchor=(2., 1.))


'''
Here starts the function dump. Stuff that I think is not necessary anymore.
'''
import functools
import warnings


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func


@deprecated
def fit_base_model(data, use_cma=False):
    start = [0., 0.] + [1.] * 10
    return fit_model(data, start, base_model, use_cma=use_cma)


@deprecated
def fit_varmodel(data, use_cma=False):
    start = [0., 1., 0.] + [1.] * 10
    return fit_model(data, start, variance_model, use_cma=use_cma)


@deprecated
def fit_model(data, start, func, use_cma=False):
    contrast = get_contrast(data)
    resp = (data.response + 1) / 2.
    conf = data.conf_0index.values

    err_function = lambda x: err(list(x),
                                 contrast=contrast, confidence=conf, responses=resp.values,
                                 func=func)

    if use_cma:
        x = cma.fmin(err_function, start, 1.5, restarts=5, options={
            'verbose': -9})[0]
        L = err_function(x)
        return x, L
    else:
        out = minimize(err_function,
                       start, method='Nelder-Mead',
                       options={'maxiter': 5000})
        return out.x, out.fun


@deprecated
def err(modelargs, **kwargs):
    '''
    Kwargs needs to contain:
        contrast, confidence, responses
    '''
    if 'func' in list(kwargs.keys()):
        model = kwargs['func']

    y = model(*modelargs, contrast=kwargs['contrast'])
    if np.any(np.isnan(y)):
        print('NAN prediction')
        return np.inf
    yy = expit(y)
    yy = np.maximum(yy, np.finfo(float).eps)
    yy = np.minimum(yy, 1 - np.finfo(float).eps)
    v = -np.sum((1 - kwargs['responses']) *
                np.log(1 - yy) + kwargs['responses'] * np.log(yy))
    if np.isnan(v):
        print(modelargs, v)
        1 / 0
    return v  # +v



def sigmoid_transformer(x, a, b):
    '''
    x is in range (-1, 1)
    '''

    b = erf(b)
    if a >= 0:
        y = erf(abs(a) * (x + b))
        y0 = erf(abs(a) * (0 + b))
        y -= y.min()
        return 2 * ((y / y.max()) - 0.5), 2 * ((y0 / y.max()) - 0.5),
    if a < 0:
        xx = np.linspace(-1, 1, 200)
        y = erf(-a * (xx + b))
        y -= y.min()
        y = 2 * ((y / y.max()) - 0.5)
        spl = interp1d(y, xx, kind='linear')
        return spl(x), spl(0)


@deprecated
def variance_model_old(offset, varslope, a, *tslope, **kwargs):
    contrast = kwargs['contrast']

    # Transform contrast and remove ref such that 0 is ideal decision boundary
    transformed = transformer(contrast, a) - transformer(0.5, a)
    weights = np.array(tslope)
    transformed = weights[np.newaxis, :] * transformed

    stdc = transformed.std(1)
    stdc = (stdc) / stdc.std()
    y = transformed.sum(1) + varslope * stdc + offset

    return y


@deprecated
def ph_power_transformer(x, a):
    '''
    x is in range (0, 1), maps to (0, 1)
    a is in range(-inf, inf)
    '''
    return x**pm.math.exp(a)


@deprecated
def conf_err(modelargs, **kwargs):
    '''
    Compute likelihood that confidence response is sample from a multinomial
    distribution.


    Kwargs needs to contain:
        contrast, confidence, responses, func
    '''
    if 'func' in list(kwargs.keys()):
        model = kwargs['func']

    y = model(*modelargs, contrast=kwargs['contrast'])
    yy = expit(y)
    yy = np.maximum(yy, np.finfo(float).eps)
    yy = np.minimum(yy, 1 - np.finfo(float).eps)
    # yy is the probability of making a 'yes' choice
    # Now map to multinomial distribution.
    conf_width = 1.0
    ps = np.array([0, 0.25, 0.5, 0.75, 1])[:, np.newaxis]
    p = np.diff(norm.cdf(logit(ps), logit(yy), conf_width), axis=0)
    # r are the confidence responses in [0, 1, 2, 3], i.e. [high no, low no,
    # low yes, high yes]
    r = (kwargs['confidence'] + 1.) * kwargs['responses'] + 2
    r[r > 2] -= 1
    v = -np.log(p[r]).sum()
    return v  # +v
