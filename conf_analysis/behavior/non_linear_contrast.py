import pylab as plt
import numpy as np
import patsy
import statsmodels.api as sm
import pandas as pd
from conf_analysis.behavior import empirical


parameters = {'djh_V1': {'a': 0.99, 'sigma': 2.58, 'p': 0.33, 'q': 1.55},
              'djh_V2d': {'a': 0.82, 'sigma': 2.31, 'p': 0.34, 'q': 1.62},
              'djh_V3d': {'a': 0.67, 'sigma': 2.35, 'p': 0.31, 'q': 1.66},
              'djh_V3A': {'a': 0.48, 'sigma': 1.23, 'p': 0.22, 'q': 5.19}}


def make_R(a=1, p=0.2, q=1.5, sigma=2.5):
    def foo(C):
        C = np.minimum(np.maximum(C, 0.001), 0.999)
        f = lambda C: (a * (100 * C)**(p + q)) / ((100 * C)**q + sigma**q)
        y = f(C)
        return (y - f(0)) / (f(0.5) - f(0)) - 0.5
    return foo

R = make_R(**parameters['djh_V1'])


def plot_wotrans(data, R, bins, fit_cutoff=0.3):
    '''
    Plot optimal model behavior without adjusting for CTF bias.
    '''
    transform(R, data)
    data.loc[:, 'true_mc'] = (data.mc > 0.5).astype(int)
    log_res, data_results = empirical.fit_logistic(
        data, 'R ~ mc + stdc', summary=False)

    log_res, results = empirical.fit_logistic(
        data, 'true_mc ~ trans_stdc + trans_mc', summary=False)
    data = data.copy()
    data.loc[:, 'R'] = data.trans_mc > R(0.5)
    data.loc[:, 'response'] = 2 * ((data.trans_mc > R(0.5)) - 0.5)
    print((results.params))
    log_res, results = empirical.fit_logistic(data, 'R~mc+stdc', summary=False)
    stuff = plot_model(data, results, bins=bins,
                       cmap='RdBu_r', plot_hyperplane=True)
    plot_model(data, data_results, bins=bins, cmap='RdBu_r',
               hyperplane_only=True, plot_hyperplane=True)
    return stuff


def t(func, X):
    mc = func(X).mean(1)
    return mc


def err(X, gamma):
    mc = (X**gamma).mean(1)
    responses = mc < (0.5**gamma)
    return responses


def fit_gamma(data, gs=np.linspace(-0.01, 10, 1000)):
    cvals = np.vstack(data.contrast_probe)
    r = []

    for g in gs:
        rr = err(cvals, g)
        r.append(np.mean(rr == (data.response < 0)))
    return gs, r


def plot(data, R, bins, fit_cutoff=0.3,
         expected_mean_bins=[np.linspace(
             0.01, 0.7, 31), np.linspace(2.8, 4.2, 101)],
         expected_mean_data=None):
    dt = transform(R, data)
    log_res, results = empirical.fit_logistic(
        data, 'R ~ trans_mc + trans_stdc', summary=False)
    print((results.params))
    plot_model(data, results, bins=bins, cmap='RdBu_r', mc_field='trans_mc',
               stdc_field='trans_stdc', plot_hyperplane=True)
    # if expected_mean_data is None:
    #    expected_mean_data = data
    #expected_mean(expected_mean_data, expected_mean_bins)
    plt.plot([R(0.5), R(0.5)], plt.ylim(), 'k--')


def transform(func, df):
    mc = np.array([np.mean(func(k)) for k in df.contrast_probe.values])
    print((np.isnan(mc).sum()))
    stdc = np.array([np.std(func(k)) for k in df.contrast_probe.values])
    df.loc[:, 'trans_mc'] = mc
    df.loc[:, 'trans_stdc'] = stdc
    return df


def fit_logistic(df, formula, summary=True):
    y, X = patsy.dmatrices(formula, df, return_type='dataframe')
    log_res = sm.GLM(y, X, family=sm.families.Binomial())
    results = log_res.fit(disp=False)
    if summary:
        print((results.summary()))
    return log_res, results


def plot_ratio(df, bins=[np.linspace(0, .25, 100), np.linspace(0, 1, 100)],
               alpha=1, cmap=None, mc_field='mc', stdc_field='stdc', response_field='response'):
    C, M = plt.meshgrid(*bins)
    resp1 = plt.histogram2d(df[df.loc[:, response_field] == 1].loc[:, stdc_field].values,
                            df[df.loc[:, response_field] == 1].loc[:, mc_field].values, bins=bins)[0] + 1
    #resp1[resp1==1] = np.nan
    resp2 = plt.histogram2d(df[df.loc[:, response_field] == -1].loc[:, stdc_field].values,
                            df[df.loc[:, response_field] == -1].loc[:, mc_field].values, bins=bins)[0] + 1
    #resp2[resp2==1] = np.nan
    resp1 = resp1.astype(float) / np.nansum(resp1)
    resp2 = resp2.astype(float) / np.nansum(resp2)
    plane = np.log(resp1 / resp2)
    #plane[plane==1] = np.nan
    plt.pcolormesh(bins[1], bins[0], np.ma.masked_invalid(
        plane), cmap=cmap, vmin=-2.4, vmax=2.4)


def plot_model(df, model, bins=[np.linspace(0, .25, 100), np.linspace(0, 1, 100)],
               hyperplane_only=False, alpha=1, cmap=None,
               mc_field='mc', stdc_field='stdc', response_field='response',
               plot_hyperplane=True):
    C, M = plt.meshgrid(*bins)
    resp1 = plt.histogram2d(df[df.loc[:, response_field] == 1].loc[:, stdc_field].values,
                            df[df.loc[:, response_field] == 1].loc[:, mc_field].values, bins=bins)[0] + 1
    #resp1[resp1==1] = np.nan
    resp2 = plt.histogram2d(df[df.loc[:, response_field] == -1].loc[:, stdc_field].values,
                            df[df.loc[:, response_field] == -1].loc[:, mc_field].values, bins=bins)[0] + 1
    #resp2[resp2==1] = np.nan
    resp1 = resp1.astype(float) / np.nansum(resp1)
    resp2 = resp2.astype(float) / np.nansum(resp2)

    p = model.predict(
        np.vstack([M.ravel(), C.ravel(), 0 * np.ones(M.shape).ravel()]).T)
    p = p.reshape(M.shape)

    decision = lambda x: - \
        (model.params[mc_field] * x + model.params.Intercept) / \
        model.params[stdc_field]
    if not hyperplane_only:

        plane = np.log(resp1 / resp2)
        #plane[plane==1] = np.nan
        plt.pcolormesh(bins[1], bins[0], np.ma.masked_invalid(
            plane), cmap=cmap, vmin=-2.4, vmax=2.4)

    mind, maxd = plt.xlim()
    plt.ylim(bins[0][0], bins[0][-1])
    plt.xlim(bins[1][0], bins[1][-1])
    if plot_hyperplane:
        plt.plot([mind, maxd], [decision(mind), decision(maxd)],
                 'k', lw=2, alpha=alpha)
    plt.plot([0, 0], [bins[0][0], bins[0][-1]], 'k--', lw=2)
    return bins


def expected_mean(data, bins, levels=[0.5]):
    resp1 = plt.histogram2d(data.trans_stdc.values, data.trans_mc.values, weights=data.mc,
                            bins=bins)[0]
    resp_count = plt.histogram2d(data.trans_stdc.values, data.trans_mc.values,
                                 bins=bins)[0]
    expected = resp1 / resp_count.astype(float)
    CS = plt.contour(center(bins[1]), center(
        bins[0]), np.ma.masked_invalid(expected), levels, colors='m')
    zc = CS.collections[0]
    plt.setp(zc, linewidth=2.5)
    return expected


def center(bins):
    return [low + (high - low) / 2. for low, high in zip(bins[0:-1], bins[1:])]
