"""
This module models gamma band responses to individual samples by assuming that
each response follows a parameterized function (normals for now). The
amplitude of each response is assumed to be modulated by contrast.
"""

import numpy as np
import pandas as pd

from math import gamma

from itertools import product

from . import srplots
from ..behavior import individual_sample_model as ism
from numba import jit
from numba import vectorize
from scipy.optimize import least_squares

from scipy.interpolate import interp1d
from scipy.special import erf

from scipy.stats import multivariate_normal as mvnorm
from scipy.stats import norm

from conf_analysis.behavior import metadata
from joblib import Memory


import socket

if "lisa.surfsara" in socket.gethostname():
    cachedir = "/home/nwilming"
else:
    cachedir = metadata.cachedir

memory = Memory(cachedir=metadata.cachedir)


@vectorize(["f8(f8)", "f4(f4)"])
def expit(x):
    if x > 0:
        x = np.exp(x)
        return x / (1 + x)
    else:
        return 1 / (1 + np.exp(-x))


@jit
def gauss(x, mean, sigma, amplitude=1):
    return np.exp(-(x - mean) ** 2 / sigma).reshape((-1, 1)) * amplitude.reshape(
        (1, -1)
    )


@jit
def power_transformer(x, a):
    """
    x is in range (0, 1), maps to (0, 1)
    a is in range(-inf, inf)
    """
    y = x ** np.exp(a)
    return y


@jit(nopython=True)
def sigmoidal_old(contrast, slope, offset):
    """
    Sigmoidal transform of contrast
    """
    y = expit((slope * (contrast - 0.5 + offset))) - 0.5
    y = y / (expit((slope * (1 - 0.5 + offset))) - 0.5)
    return y


@jit(nopython=True)
def sigmoidal(contrast, slope, offset):
    """
    Sigmoidal transform of contrast
    """
    y = expit((slope * (contrast - 0.5 + offset))) - 0.5
    return y / (expit((slope * (1 - 0.5))) - 0.5)


def sigmoid_transformer(x, amplitude, offset, a, b):
    """
    x is in range (-1, 1)
    """

    b = erf(b)
    if a >= 0:
        y = erf(abs(a) * (x + b))
        y -= y.min()
        y = 2 * ((y / y.max()) - 0.5)
    if a < 0:
        xx = np.linspace(-1, 1, 200)
        y = erf(-a * (xx + b))
        y -= y.min()
        y = 2 * ((y / y.max()) - 0.5)
        spl = interp1d(y, xx, kind="linear")
        y = spl(x)
    return amplitude * y + offset


def spitzer_exp(contrast, kappa, offset):
    """
    Kappa is parameterized such that 0 is a linear relationship
    Negative kappa is compression
    """
    term = (2 * (contrast - 0.5)) + offset
    return (np.abs(term) ** kappa) * term / np.abs(term)


def fit(predict, data, x0, bounds):
    err = lambda x: (data - predict(x)).ravel()

    return least_squares(err, x0, loss="soft_l1", bounds=bounds)


def fit2(predict, data, contrast, x0, bounds):
    err = lambda x: (data - predict(contrast, *x)).ravel()

    return least_squares(err, x0, loss="soft_l1")


"""
Linear model with temporal differences: lmtd
Contrast dependence is simply linear!
"""


@jit(nopython=True)
def lmtd_predict(
    time,
    contrast,
    offset=None,
    latency=0.2,
    std=0.05,
    amplitude_parameters=1,
    diff_latency=0.2,
    diff_std=0.05,
    diff_amplitude_parameters=1,
):
    if len(contrast.shape) < 2:
        raise RuntimeError("Contrast array needs to be of dimension trials:10")
    output = time.reshape((-1, 1)) * np.zeros((1, contrast.shape[0]))

    for ii, onset in enumerate(np.arange(0, 1, 0.1)):
        output += gauss(
            time, onset + latency, std, contrast[:, ii] * amplitude_parameters
        )

    cdiff = np.diff(contrast, 1)
    onsets = np.arange(0.1, 1, 0.1)
    for ii, onset in enumerate(onsets):
        output += gauss(
            time,
            onset + latency + diff_latency,
            diff_std,
            cdiff[:, ii] * diff_amplitude_parameters,
        )
    return output + offset.reshape((-1, 1))


def lmtd_params2dict(time, x):
    tl = len(time)
    return {
        "offset": x[0:tl],
        "latency": x[tl],
        "std": x[tl + 1],
        "amplitude_parameters": x[tl + 2],
        "diff_latency": x[tl + 3],
        "diff_std": x[tl + 4],
        "diff_amplitude_parameters": x[tl + 5],
    }


def lmtd_fit(data, contrast, time):
    """
    Use non-linear least squares to find parameters.
    """
    x0 = np.concatenate((data.mean(1), [0.15, 0.001, 1, 0.001, 0.001, 0.5]))

    bounds = (
        [-np.inf] * len(time) + [0.0, 0.0, -np.inf, -0.1, 0.0, -np.inf],
        [+np.inf] * len(time) + [0.5, 0.1, +np.inf, +0.1, 0.1, +np.inf],
    )

    yp = lambda x: lmtd_predict(time, contrast, **lmtd_params2dict(time, x))
    out = fit(yp, data, x0, bounds)
    return lmtd_params2dict(time, out["x"])


"""
Linear model with temporal differences and transformed
contrast: nltd
Contrast dependence is non-linear!
"""


@jit(nopython=True)
def nltd_predict(
    time,
    contrast,
    offset=None,
    latency=0.2,
    std=0.05,
    non_linearity=1,
    amplitude_parameters=1,
    diff_latency=0.2,
    diff_std=0.05,
    diff_amplitude_parameters=1,
    **kw
):
    if len(contrast.shape) < 2:
        raise RuntimeError("Contrast array needs to be of dimension trials:10")
    output = time.reshape((-1, 1)) * np.zeros((1, contrast.shape[0]))

    contrast = power_transformer(contrast, non_linearity) - power_transformer(
        0.5, non_linearity
    )  # Now contrast of 0.5 -> 0

    for ii, onset in enumerate(np.arange(0, 1, 0.1)):
        output += gauss(
            time, onset + latency, std, contrast[:, ii] * amplitude_parameters
        )

    cdiff = np.diff(contrast, 1)
    onsets = np.arange(0.1, 1, 0.1)
    for ii, onset in enumerate(onsets):
        output += gauss(
            time,
            onset + latency + diff_latency,
            diff_std,
            cdiff[:, ii] * diff_amplitude_parameters,
        )

    return output + offset.reshape((-1, 1))


def nltd_params2dict(time, x):
    tl = len(time)
    return {
        "offset": x[0:tl],
        "latency": x[tl],
        "std": x[tl + 1],
        "non_linearity": x[tl + 2],
        "amplitude_parameters": x[tl + 3],
        "diff_latency": x[tl + 4],
        "diff_std": x[tl + 5],
        "diff_amplitude_parameters": x[tl + 6],
    }


def nltd_fit(data, contrast, time):
    """
    Use non-linear least squares to find parameters.
    """
    from scipy.optimize import least_squares

    x0 = np.concatenate((data.mean(1), [0.15, 0.001, 1.0, 0.0, 0.001, 0.001, 0.5]))

    bounds = (
        [-np.inf] * len(time) + [0.0, 0.0, -1, -100, -0.1, 0.0, -np.inf],
        [+np.inf] * len(time) + [0.5, 0.1, +1, +100, +0.1, 0.1, +np.inf],
    )

    yp = lambda x: nltd_predict(time, contrast, **nltd_params2dict(time, x))
    err = lambda x: (data - yp(x)).ravel()

    out = least_squares(err, x0, loss="soft_l1", bounds=bounds)
    return nltd_params2dict(time, out["x"])


"""
Linear model with temporal differences and sigmoid
contrast transform: sltd
The contrast dependence is non-linear.
"""


def sltd_predict(
    time,
    contrast,
    offset=None,
    latency=0.2,
    std=0.05,
    slope=1.0,
    sigmoid_offset=0.0,
    amplitude_parameters=1,
    diff_latency=0.2,
    diff_std=0.05,
    diff_amplitude_parameters=1,
):
    if len(contrast.shape) < 2:
        raise RuntimeError("Contrast array needs to be of dimension trials:10")
    contrast = sigmoidal(contrast, slope, sigmoid_offset)
    output = time.reshape((-1, 1)) * np.zeros((1, contrast.shape[0]))

    for ii, onset in enumerate(np.arange(0, 1, 0.1)):
        output += gauss(
            time, onset + latency, std, contrast[:, ii] * amplitude_parameters
        )

    cdiff = np.diff(contrast, 1)
    onsets = np.arange(0.1, 1, 0.1)
    for ii, onset in enumerate(onsets):
        output += gauss(
            time,
            onset + latency + diff_latency,
            diff_std,
            cdiff[:, ii] * diff_amplitude_parameters,
        )
    if offset is not None:
        output += offset.reshape((-1, 1))
    return output


def sltd_params2dict(time, x, offset=True):
    params = {}
    if len(x) != int(offset) * len(time) + 8:
        raise RuntimeError(
            "Number of parameters is wrong. Is %i, should be %i"
            % (len(x), len(time) + 8)
        )
    if offset:

        tl = len(time)
        params["offset"] = x[0:tl]
    else:
        tl = 0
        params["offset"] = None

    params.update(
        {
            "latency": x[tl],
            "std": x[tl + 1],
            "slope": x[tl + 2],
            "sigmoid_offset": x[tl + 3],
            "amplitude_parameters": x[tl + 4],
            "diff_latency": x[tl + 5],
            "diff_std": x[tl + 6],
            "diff_amplitude_parameters": x[tl + 7],
        }
    )
    return params


def sltd_fit(data, contrast, time, offset=True):
    """
    Use non-linear least squares to find parameters.
    """
    from scipy.optimize import least_squares

    if offset:
        x0 = np.concatenate(
            (data.mean(1), [0.15, 0.001, 1.0, 0.0, 0.0, 0.001, 0.001, 0.5])
        )
        # Offset, latency, std, slope, sigmoid_offset, ...
        bounds = (
            [-np.inf] * len(time) + [0.0, 0.0, 0, -0.5, -np.inf, -0.1, 0.0, -np.inf],
            [+np.inf] * len(time) + [0.5, 0.1, +10, +0.5, +np.inf, +0.1, 0.1, +np.inf],
        )
    else:
        x0 = np.array([0.15, 0.001, 1.0, 0.0, 0.0, 0.001, 0.001, 0.5])
        data = data - data.mean(0)
        bounds = (
            [0.0, 0.0, -10, -0.5, -np.inf, -0.1, 0.0, -np.inf],
            [0.5, 0.1, +10, +0.5, +np.inf, +0.1, 0.1, +np.inf],
        )

    yp = lambda x: sltd_predict(
        time, contrast, **sltd_params2dict(time, x, offset=offset)
    )
    err = lambda x: (data - yp(x)).ravel()

    out = least_squares(err, x0, loss="soft_l1", bounds=bounds)
    return sltd_params2dict(time, out["x"], offset=offset)


"""
Linear model with temporal differences and Spitzer et al.
non-linearities: spitzer
The contrast dependence is non-linear and can follow a sigmoid
or inverted sigmoid.
"""


def spitzer_predict(
    time,
    contrast,
    offset=None,
    latency=0.2,
    std=0.05,
    kappa=1.0,
    sigmoid_offset=0.0,
    amplitude_parameters=1,
    diff_latency=0.2,
    diff_std=0.05,
    diff_amplitude_parameters=1,
):
    if len(contrast.shape) < 2:
        raise RuntimeError("Contrast array needs to be of dimension trials:10")
    contrast = spitzer_exp(contrast, kappa, sigmoid_offset)
    output = time.reshape((-1, 1)) * np.zeros((1, contrast.shape[0]))

    for ii, onset in enumerate(np.arange(0, 1, 0.1)):
        output += gauss(
            time, onset + latency, std, contrast[:, ii] * amplitude_parameters
        )

    cdiff = np.diff(contrast, 1)
    onsets = np.arange(0.1, 1, 0.1)
    for ii, onset in enumerate(onsets):
        output += gauss(
            time,
            onset + latency + diff_latency,
            diff_std,
            cdiff[:, ii] * diff_amplitude_parameters,
        )
    if offset is not None:
        output += offset.reshape((-1, 1))
    return output


def spitzer_params2dict(time, x, offset=True):
    params = {}
    if len(x) != int(offset) * len(time) + 8:
        raise RuntimeError(
            "Number of parameters is wrong. Is %i, should be %i"
            % (len(x), len(time) + 8)
        )
    if offset:

        tl = len(time)
        params["offset"] = x[0:tl]
    else:
        tl = 0
        params["offset"] = None

    params.update(
        {
            "latency": x[tl],
            "std": x[tl + 1],
            "kappa": x[tl + 2],
            "sigmoid_offset": x[tl + 3],
            "amplitude_parameters": x[tl + 4],
            "diff_latency": x[tl + 5],
            "diff_std": x[tl + 6],
            "diff_amplitude_parameters": x[tl + 7],
        }
    )
    return params


def spitzer_fit(data, contrast, time, offset=True):
    """
    Use non-linear least squares to find parameters.
    """
    from scipy.optimize import least_squares

    if offset:
        x0 = np.concatenate(
            (data.mean(1), [0.15, 0.001, 1.0, 0.0, 0.0, 0.001, 0.001, 0.5])
        )
        # Offset, latency, std, slope, sigmoid_offset, ...
        bounds = (
            [-np.inf] * len(time) + [0.0, 0.0, 0, -0.5, -np.inf, -0.1, 0.0, -np.inf],
            [+np.inf] * len(time) + [0.5, 0.1, +10, +0.5, +np.inf, +0.1, 0.1, +np.inf],
        )
    else:
        x0 = np.array([0.15, 0.001, 1.0, 0.0, 0.0, 0.001, 0.001, 0.5])
        data = data - data.mean(0)
        bounds = (
            [0.0, 0.0, -10, -0.5, -np.inf, -0.1, 0.0, -np.inf],
            [0.5, 0.1, +10, +0.5, +np.inf, +0.1, 0.1, +np.inf],
        )

    yp = lambda x: sltd_predict(
        time, contrast, **sltd_params2dict(time, x, offset=offset)
    )
    err = lambda x: (data - yp(x)).ravel()

    out = least_squares(err, x0, loss="soft_l1", bounds=bounds)
    return sltd_params2dict(time, out["x"], offset=offset)


"""
Analysis functions
"""


@memory.cache
def make_sub_data(subject, area, F=55, log=True):
    df, meta = srplots.get_power(subject, decim=3, F=F)
    cvals = ism.get_contrast(meta)
    data = make_data(df, area=area, log=log)
    return data, cvals


def make_data(df, area="V1-lh", log=False):
    data = pd.pivot_table(df, index="trial", columns="time", values=area).loc[
        :, -0.2:1.3
    ]
    if log:
        data = np.log(data)
    base = data.loc[:, -0.2:0].mean(1)
    bases = data.loc[:, -0.2:0].std(1)
    return data.subtract(base, 0).div(bases, 0)


@memory.cache
def subject_fit(subject, F, area, fit_func="stld"):
    fit_map = {"stld": sltd_fit, "nltd": nltd_fit, "lmtd": lmtd_fit}
    data, contrast = make_sub_data(subject, area, F=F)
    time = data.columns.values
    params = fit_map[fit_func](data.values.T, contrast, time)
    return data, contrast, params


def subject_sa(
    subject,
    F=[40, 45, 50, 55, 60, 65, 70],
    areas=["V1-lh", "V2-lh", "V3-lh", "V1-rh", "V2-rh", "V3-rh"],
    window=0.01,
    remove_overlap=True,
):
    """
    Extraxt single trial power estimates.
    """
    acc_real, acc_pred = [], []
    for f, area in product(F, areas):
        if not subject_fit.is_cached(subject, f, area):
            print("Skipping", subject, f, area)
            continue
        data, contrast, params = subject_fit(subject, f, area)
        sa = sample_aligned(
            data,
            contrast,
            params,
            window=window,
            predict_func=sltd_predict,
            remove_overlap=remove_overlap,
        )

        sa.loc[:, "subject"] = subject
        sa.loc[:, "F"] = f
        sa.loc[:, "area"] = area
        acc_real.append(sa)
        predicted = sltd_predict(data.columns.values, contrast, **params).T
        predicted = pd.DataFrame(
            data, index=np.arange(data.shape[0]), columns=data.columns.values
        )

        sa = sample_aligned(
            predicted,
            contrast,
            params,
            window=window,
            predict_func=sltd_predict,
            remove_overlap=remove_overlap,
        )
        sa.loc[:, "subject"] = subject
        sa.loc[:, "F"] = f
        sa.loc[:, "area"] = area
        acc_pred.append(sa)
    return pd.concat(acc_real), pd.concat(acc_pred)


def make_all_sub_sa():
    out_sa, out_sp = [], []
    for subject in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        sa, sp = subject_sa(subject)
        out_sa.append(sa)
        out_sp.append(sp)
    return pd.concat(out_sa), pd.concat(out_sp)


def sample_aligned(
    data, contrast, params, window=0.01, remove_overlap=True, predict_func=sltd_predict
):
    """
    Produce sample aligned power values, but remove effec from previous sample
    """

    latency = params["latency"]
    output = []
    cnt = 0
    for sample, onset in enumerate(np.arange(0, 1, 0.1) + latency):
        if remove_overlap:
            con = contrast.copy()
            con[:, sample] = 0.5
            predicted = predict_func(data.columns.values, con, **params).T
            predicted = pd.DataFrame(predicted, columns=data.columns, index=data.index)
            vs = (data - predicted).loc[:, onset - window : onset + window].mean(1)
        else:
            vs = data.loc[:, onset - window : onset + window].mean(1)
        for trial, cc, d in zip(vs.index.values, contrast[:, sample], vs):
            output.append(
                {
                    "sample": sample,
                    "power": d,
                    "contrast": cc,
                    "trial": trial,
                    "sample_id": cnt,
                }
            )
            cnt += 1
    return pd.DataFrame(output)


def get_average_contrast(
    sa, by=["subject", "area", "F"], centers=np.linspace(0, 1, 10), width=0.2
):
    dy = sa.groupby(by).apply(lambda x: contrast_integrated_averages(x, centers, width))
    return dy


def contrast_integrated_averages(sa, centers=np.linspace(0.1, 0.9, 5), width=0.2):
    sa = sa.reset_index()
    rows = []
    contrast = sa.loc[:, "contrast"].values
    w = width / 2.0
    for center in centers:
        idx = ((center - w) < contrast) & (contrast < (center + w))
        print(sum(idx))
        r = sa.loc[idx, ("contrast", "power")].mean()
        r.loc["contrast"] = center
        rows.append(r)
    return pd.concat(rows, 1).T


def example_plot():
    from pylab import plot, subplot, ylim, xlabel, ylabel
    import seaborn as sns

    c = np.linspace(0, 1, 100)
    subplot(1, 3, 1)
    color = sns.color_palette("Reds", n_colors=10)
    for ii, slope in enumerate(np.arange(1, 11, 1)):
        plot(c, sigmoidal(c, slope, 0), color=color[ii])
        ylim([-1, 1])
        xlabel("contrast")
        ylabel(r"$\Delta Power$")
    subplot(1, 3, 2)
    color = sns.color_palette("Greens", n_colors=10)
    for ii, amplitude in enumerate(np.linspace(0, 1, 10)):
        plot(c, amplitude * sigmoidal(c, 2.5, 0), color=color[ii])
        ylim([-1, 1])
        xlabel("contrast")
    subplot(1, 3, 3)
    color = sns.color_palette("Blues", n_colors=10)
    for ii, offset in enumerate(np.linspace(-0.5, 0.5, 10)):
        plot(c, 1 * sigmoidal(c, 5, offset), color=color[ii])
        ylim([-1, 1])
        xlabel("contrast")

    sns.despine(offset=5)


def plot_fit_params(parameters):
    import seaborn as sns

    ps = (
        parameters.set_index(["F", "area", "subject", "hemisphere"])
        .loc[:, ["slope", "amplitude_parameters", "sigmoid_offset", "latency"]]
        .stack()
        .reset_index()
    )
    ps.columns = ["F", "area", "subject", "hemisphere", "parameter", "value"]
    g = sns.FacetGrid(
        ps,
        col="hemisphere",
        row="parameter",
        hue="area",
        sharey=False,
        sharex=True,
        size=2,
        aspect=2.5,
    )
    g.map(sns.pointplot, "F", "value")
    g.axes[0, 0].set_ylim([0, 11])
    g.axes[0, 1].set_ylim([0, 11])

    g.axes[1, 0].set_ylim([-0.1, 0.6])
    g.axes[1, 1].set_ylim([-0.1, 0.6])

    g.axes[2, 0].set_ylim([-0.4, 0.3])
    g.axes[2, 1].set_ylim([-0.4, 0.3])

    g.axes[3, 0].set_ylim([0.0, 0.3])
    g.axes[3, 1].set_ylim([0.0, 0.3])


"""
Bayesian fits
"""


def mv_model(power, contrast):
    """
    Define a model that predicts contrast from power. Two step approach:
    First fit likelihood to observe a power value based on contrast.

    Power is num_freqs x trials

    """
    import numpy as np

    num_freqs = power.shape[1]
    means = power.mean(0)
    stds = power.std(0)
    power = (power - means[np.newaxis, :]) / stds[np.newaxis, :]

    import numpy as np
    import theano.tensor as tt
    import pymc3 as pm

    with pm.Model() as model:
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
        slope = pm.Normal("slope", mu=0, sd=1, shape=(1, num_freqs))

        amplitude = BoundedNormal("amplitude", mu=1, sd=1, shape=(1, num_freqs))
        offset = pm.Normal("offset", mu=0, sd=0.1, shape=(1, num_freqs))

        # Now define contrast dependence.
        mu = amplitude * (
            pm.math.invlogit(slope * (contrast[:, np.newaxis] + offset)) - 0.5
        )
        # mu =  contrast[:, np.newaxis]*slope + offset
        # sigma = pm.HalfCauchy("sigma", beta=5, shape=10)

        packed_L = pm.LKJCholeskyCov(
            "packed_L", n=num_freqs, eta=2, sd_dist=pm.HalfCauchy.dist(beta=5)
        )
        L = pm.expand_packed_triangular(num_freqs, packed_L)
        sigma = pm.Deterministic("sigma", L.dot(L.T))
        # y_ = pm.MvNormal('obs', mu, chol=L, observed=power)
        nu = BoundedNormal("NU", mu=3, sd=20)
        y_ = pm.MvStudentT("obs", mu=mu, chol=L, nu=nu, observed=power)
        return model


def mv_crf_model(power, contrast):
    """
    Fit contrast response functions with two independent exponents.

    The CRF has four parameters:
        m - magnitude
        p - exponent in numerator
        q - added to p in denominator
        c - point of half contrast.

    Contrast needs will be scaled to [0..100], but is expected to be
    in [0..1]
    """
    num_freqs = power.shape[1]
    import numpy as np
    import pymc3 as pm

    contrast = contrast + 0.5

    rmin = power[contrast < 0.2, :].mean(0)
    rmax = power[contrast > 0.8, :].mean(0) - rmin

    with pm.Model() as model:
        BoundedNormal = pm.Bound(pm.Normal, lower=0.01)
        Rmax = pm.Normal("Rmax", mu=rmax, sd=1, shape=(1, num_freqs))
        Rmin = pm.Normal("Rmin", mu=rmin, sd=1, shape=(1, num_freqs))
        p = BoundedNormal("P", mu=0, sd=10, shape=(1, num_freqs))
        c50 = BoundedNormal("c50", mu=0.5, sd=0.1, shape=(1, num_freqs))

        # Now define contrast dependence.
        mu = crf_scaled(contrast, Rmax, Rmin, p, c50)

        packed_L = pm.LKJCholeskyCov(
            "packed_L", n=num_freqs, eta=10.0, sd_dist=pm.HalfCauchy.dist(5.0)
        )
        L = pm.expand_packed_triangular(num_freqs, packed_L)
        sigma = pm.Deterministic("sigma", L.dot(L.T))
        # nu = BoundedNormal("NU", mu=3, sd=20)
        # y_ = pm.MvStudentT("obs", mu=mu, chol=L, nu=nu, observed=power)
        y_ = pm.MvNormal("obs", mu=mu, chol=L, observed=power)
        return model


def mv_crf_sample_model(power, contrast, sample):
    """
    Fit contrast response functions with two independent exponents.

    Fit slope as a function of sample at the same time.
    """
    num_freqs = power.shape[1]

    import numpy as np
    import pymc3 as pm

    rmin = power[contrast < 0.2, :].mean(0)
    rmax = power[contrast > 0.8, :].mean(0) - rmin

    with pm.Model() as model:
        BoundedNormal = pm.Bound(pm.Normal, lower=0.01)
        Rmax = pm.Normal(
            "Rmax", mu=rmax, sd=1, shape=(1, num_freqs)
        )  # Same across samples
        Rmin = pm.Normal(
            "Rmin", mu=rmin, sd=1, shape=(1, num_freqs)
        )  # Same across sample

        # Varies as function of sample
        p = BoundedNormal("P", mu=0, sd=15, shape=(num_freqs,))
        c50 = BoundedNormal("c50", mu=0.5, sd=0.1, shape=(num_freqs,))

        p_smpl = BoundedNormal(
            "P_sample", mu=p[np.newaxis, :], sd=1, shape=(10, num_freqs)
        )
        c50_smpl = BoundedNormal(
            "c50_sample", mu=c50[np.newaxis, :], sd=0.1, shape=(10, num_freqs)
        )

        # Now define contrast dependence.
        mu = crf_scaled(contrast, Rmax, Rmin, p_smpl[sample, :], c50_smpl[sample, :])

        packed_L = pm.LKJCholeskyCov(
            "packed_L", n=num_freqs, eta=100.0, sd_dist=pm.HalfCauchy.dist(5.0)
        )
        L = pm.expand_packed_triangular(num_freqs, packed_L)
        sigma = pm.Deterministic("sigma", L.dot(L.T))
        nu = BoundedNormal("NU", mu=3, sd=20)
        y_ = pm.MvStudentT("obs", mu=mu, chol=L, nu=nu, observed=power)
        return model


def crf_scaled(contrast, Rmax, Rmin, p, c50):
    return (
        Rmax
        * (
            contrast[:, np.newaxis] ** (p)
            / ((contrast[:, np.newaxis] ** p) + (c50 ** p))
        )
        + Rmin
    )


def sim_model_crf(cov=None):
    """
    Generate a ddict with simulated data.
    """
    nfreqs = 5
    nsubs = 15

    P_pop = np.random.rand(nfreqs) * 10
    cfifty_pop = 0.5 + np.random.randn(nfreqs) * 0.1
    cfifty_pop[cfifty_pop < 0] = 0
    cfifty_pop[cfifty_pop > 1] = 1

    P = np.abs(np.random.randn(nsubs, nfreqs) * 5 + P_pop)
    cfifty = np.random.randn(nsubs, nfreqs) * 0.05 + cfifty_pop
    cfifty[cfifty < 0] = 0
    cfifty[cfifty > 1] = 1

    Rmax = np.random.rand(nsubs, nfreqs) * 10
    Rmin = -np.random.rand(nsubs, nfreqs) * 5

    contrasts = np.random.randn(1000) * 0.1 + 0.5
    contrasts[contrasts < 0] = 0
    contrasts[contrasts > 1] = 1
    mus = []
    subject = []
    CC = []
    for sub in range(nsubs):
        mu = np.ones((len(contrasts), nfreqs))
        for freq in range(nfreqs):
            mu[:, freq] = crf_scaled(
                contrasts,
                Rmax[sub, freq],
                Rmin[sub, freq],
                P[sub, freq],
                cfifty[sub, freq],
            ).ravel()
        subject.append(mu[:, 0] * 0 + sub)
        mus.append(mu)
        CC.append(contrasts)
    subject = np.concatenate(subject)
    mus = np.vstack(mus)
    if cov is None:
        cov = np.eye(nfreqs)
    from scipy.stats import multivariate_normal

    x = np.vstack([multivariate_normal.rvs(m, cov) for m in mus])
    d = {
        "P_true": P,
        "cfifty_true": cfifty,
        "P_pop": P_pop,
        "cfifty_pop": cfifty_pop,
        "Rmin_true": Rmin,
        "Rmax_true": Rmax,
        "N": x.shape[0],
        "nfreq": x.shape[1],
        "y": x,
        "contrast": np.concatenate(CC),
        "rmin_emp": Rmin,
        "rmax_emp": Rmax,
        "subject": subject.astype(int) + 1,
        "nsub": nsubs,
    }

    init = {
        "P_pop": np.ones(nfreqs) * 10,
        "P": np.ones((nsubs, nfreqs)) * 10,
        "cfifty_pop": np.ones(nfreqs) * 0.5,
        "cfifty": np.ones((nsubs, nfreqs)) * 0.5,
        "rmin_emp": Rmin,
        "rmax_emp": Rmax,
        "Lcorr_all": cov * 0 + 0.1,
        "tau": np.diag(cov),
    }
    return d, init


def sim_model_linear(cov=None):
    """
    Generate a ddict with simulated data.
    """
    nfreqs = 5
    nsubs = 15

    P_pop = np.random.rand(nfreqs) * 10
    P = np.abs(np.random.randn(nsubs, nfreqs) * 5 + P_pop)
    Rmin = -np.random.rand(nsubs, nfreqs) * 5
    contrasts = np.random.randn(500) * 0.1 + 0.5
    contrasts[contrasts < 0] = 0
    contrasts[contrasts > 1] = 1
    mus = []
    subject = []
    CC = []
    for sub in range(nsubs):
        mu = np.ones((len(contrasts), nfreqs))
        for freq in range(nfreqs):
            mu[:, freq] = (contrasts * P[sub, freq] - Rmin[sub, freq]).ravel()
        subject.append(mu[:, 0] * 0 + sub)
        mus.append(mu)
        CC.append(contrasts)
    subject = np.concatenate(subject)
    mus = np.vstack(mus)
    if cov is None:
        cov = np.eye(nfreqs)
    from scipy.stats import multivariate_normal

    x = np.vstack([multivariate_normal.rvs(m, cov) for m in mus])
    d = {
        "P_true": P,
        "P_pop": P_pop,
        "Rmin_true": Rmin,
        "N": x.shape[0],
        "nfreq": x.shape[1],
        "y": x,
        "contrast": np.concatenate(CC),
        "rmin_emp": Rmin,
        "subject": subject.astype(int) + 1,
        "nsub": nsubs,
    }
    return d


def prepare_ddict(subject, epoch="stimulus", latency=0.18):
    from pymeg import aggregate_sr as asr
    from conf_analysis.meg import preprocessing
    from glob import glob

    data = asr.delayed_agg(
        "/home/nwilming/conf_meg/sr_labeled/aggs/S*i-*%s%.hdf" % (subject, epoch),
        hemi="Averaged",
        cluster="vfcPrimary",
    )()
    meta = preprocessing.get_meta(subject, epoch)
    meta = meta.set_index("hash")
    time = data.columns.get_level_values("time")
    time = [time[np.argmin(np.abs(time - td))] for td in latency + np.arange(0, 1, 0.1)]
    data = data.loc[:, time]
    Xs = []
    contrasts = []
    samples = []
    for sample, tidx in enumerate(time):
        X = pd.pivot_table(data, index="trial", columns="freq", values=tidx)
        trial = X.index
        Xs.append(X)
        cvals = np.vstack(meta.loc[trial, "contrast_probe"])[:, sample]
        contrasts.append(cvals)
        samples.append((cvals * 0 + sample))
    contrasts = np.concatenate(contrasts)
    samples = np.concatenate(samples)
    Xs = np.vstack(Xs)
    Xs = (Xs - Xs.mean(0)[np.newaxis, :]) / Xs.std(0)[np.newaxis, :]
    return {
        "y": Xs,
        "contrast": contrasts,
        "sample": samples,
        "subject": samples * 0 + subject,
    }


def get_data_dict():
    meta = pd.read_hdf(
        "/home/student/n/nwilming/all_vfc_subs.hdf", "contrast"
    ).to_frame()
    agg = pd.read_hdf("/home/student/n/nwilming/all_vfc_subs.hdf", "df")
    Xs = []
    contrasts = []
    time = agg.columns.get_level_values("time")
    trials = agg.index.get_level_values("trial")
    subject = agg.index.get_level_values("subject")
    dsub = pd.DataFrame(subject, index=trials)
    dsub = dsub.loc[dsub.index.unique(), :]
    samples = []
    subjects = []
    rmins = []
    rmaxs = []
    Xsubs = []
    for subject in np.arange(1, 16):
        Xs = []
        ccs = []
        for j, i in enumerate(time):
            X = pd.pivot_table(
                agg.query("subject==%i" % subject),
                values=i,
                index="trial",
                columns="freq",
            )
            x = X.values
            x = (x - x.mean(0)[np.newaxis, :]) / x.std(0)[np.newaxis, :]
            Xs.append(x)
            contrast = meta.loc[X.index, "contrast_probe"]
            subjects.append([subject] * x.shape[0])
            cvals = np.stack(contrast)
            cvals[cvals < 0] = 0
            cvals[cvals > 1] = 1
            ccs.append(cvals[:, j])
            samples.append(cvals[:, j] * 0 + j)
        Xsub = np.vstack(Xs)
        Xsubs.append(Xsub)
        c = np.concatenate(ccs)
        contrasts.append(c)
        rmin = Xsub[c < 0.2, :].mean(0)
        rmax = Xsub[c > 0.8, :].mean(0) - rmin
        rmins.append(rmin)
        rmaxs.append(rmax)
    x = np.vstack(Xsubs)[:, :]
    c = np.concatenate(contrasts)
    s = np.concatenate(samples)
    subs = np.concatenate(subjects)
    rmin = np.vstack(rmins)
    rmax = np.vstack(rmaxs)
    nsub = len(np.unique(subs))

    def init(x):
        nfreq = x.shape[1]
        corr = np.corrcoef(x.T)

        def makend(val, shape):
            return np.ones(shape) * val

        def foo():
            """
            // Priors:
            [nsub, nfreq] Rmax;     
            [nsub, nfreq] Rmin;       
            [nfreq] P[nsub];                    
            [nfreq] cfifty[nsub];   
            [nfreq] Lcorr_all;                 
            [nfreq] tau;                            

            // Hyperpriors:
            [nfreq] P_pop;                    
            [nfreq] cfifty_pop;   
            """
            return dict(
                P=makend(1, (nsub, nfreq)),
                cfifty=makend(0.5, (nsub, nfreq)),
                Rmin=makend(-0.5, (nsub, nfreq)),
                Rmax=makend(1, (nsub, nfreq)),
                Lcorr_all=np.eye(nfreq),  # np.linalg.cholesky(corr),
                P_pop=makend(1, (nfreq,)),
                cfifty_pop=makend(0.5, (nfreq,)),
            )

        return foo

    d = {
        "N": x.shape[0],
        "nfreq": x.shape[1],
        "y": x,
        "ones": cvals[:, 0] * 0 + 1,
        "contrast": c,
        "samples": s.astype(int) + 1,
        "rmin_emp": rmin,
        "rmax_emp": rmax,
        "nsamp": 10,
        "subject": subs,
        "nsub": nsub,
    }

    return d, init(d["y"])


def ppc_figure(f, y, contrast, subject, rmin_emp, rmax_emp, plot_ppc=True):
    from conf_analysis.meg import infoflow as ifo

    Rmax = f["Rmax"].data.squeeze().mean(0)
    Rmin = f["Rmin"].data.squeeze().mean(0)
    P = f["P"].data.squeeze().mean(0)
    c50 = f["cfifty"].data.squeeze().mean(0)
    import pylab as plt

    plt.figure(figsize=(10, 10))
    for i in range(0, Rmax.shape[1]):
        plt.subplot(4, 5, i + 1)
        centers, power = ifo.cia(y[:, i], contrast, centers=np.linspace(0.15, 0.85, 21))
        for j in range(1, P.shape[0], 10):
            plt.plot(
                centers,
                crf_scaled(centers, Rmax[j, i], Rmin[j, i], P[j, i], c50[j, i]),
                "b",
                alpha=0.2,
                lw=0.1,
            )
            plt.plot(centers, power, "r")
            plt.plot(-0.1, rmin_emp[j, i], "ro")
            plt.plot(1.1, rmin_emp[j, i] + rmax_emp[j, i], "ro")
            if plot_ppc:
                plt.plot(
                    centers,
                    crf_scaled(centers, Rmax[j, i], Rmin[j, i], P[j, i], c50[j, i]),
                    "b",
                    alpha=1,
                    lw=0.5,
                )

        # plt.ylim(-0.8, 0.8)


def ppc_linear_figure(f, y, contrast, subject, nth=10):
    from conf_analysis.meg import infoflow as ifo

    Rmin = f["Rmin"].data.squeeze()
    P = f["P"].data.squeeze()
    import pylab as plt

    plt.figure(figsize=(20, 20))
    for sub in range(0, P.shape[1]):
        plt.subplot(3, 5, sub + 1)
        idx = subject == sub
        for freq in range(0, P.shape[2]):            
            centers, power = ifo.cia(
                y[idx, freq], contrast[idx], centers=np.linspace(0.15, 0.85, 21)
            )
            plt.plot(centers, power, "r")
            for i in range(0, P.shape[0], nth):
                plt.plot(centers, centers * P[i, sub, freq] + Rmin[i, sub, freq], "b", alpha=0.25, lw=0.5)
            


def mv_model_eval(contrast, power, magnitude=0, P=0, Q=1, c50=50, sigma=None, NU=1):
    mu = crf(contrast * 100, magnitude, P, Q, c50)
    return np.stack([mvstudentt(power, mu[ii], sigma, NU) for ii in range(len(mu))])


def invert(trace, X, contrast, thin=10):
    pC = pc(contrast)
    out = np.zeros((len(contrast), X.shape[0]))
    cnt = 0
    for ii in range(0, trace[list(trace.keys())[0]].shape[0], thin):
        params = dict((k, trace[k][ii, :]) for k in list(trace.keys()))
        y = mv_model_eval(contrast, X, **params)
        y = y / y.sum(0)[np.newaxis, :]
        out += y * pC
        cnt += 1.0
    out = out / cnt
    return out / out.sum(0)[np.newaxis, :]


def pc(x, t=0.05):
    y = (
        norm.pdf(x, 0.5 + t, 0.05)
        + norm.pdf(x, 0.5 + t, 0.1)
        + norm.pdf(x, 0.5 + t, 0.15)
        + norm.pdf(x, 0.5 - t, 0.05)
        + norm.pdf(x, 0.5 - t, 0.1)
        + norm.pdf(x, 0.5 - t, 0.15)
    )
    return y / y.sum()


def mvstudentt(x, mu, Sigma, df):
    """
    Multivariate t-student density. Returns the density
    of the function at points specified by x.

    input:
        x = parameter (n-d numpy array; will be forced to 2d)
        mu = mean (d dimensional numpy array)
        Sigma = scale matrix (dxd numpy array)
        df = degrees of freedom

    Edited from: http://stackoverflow.com/a/29804411/3521179
    """

    x = np.atleast_2d(x)  # requires x as 2d
    nD = Sigma.shape[0]  # dimensionality

    numerator = gamma(1.0 * (nD + df) / 2.0)

    denominator = (
        gamma(1.0 * df / 2.0)
        * np.power(df * np.pi, 1.0 * nD / 2.0)
        * np.power(np.linalg.det(Sigma), 1.0 / 2.0)
        * np.power(
            1.0
            + (1.0 / df)
            * np.diagonal(np.dot(np.dot(x - mu, np.linalg.inv(Sigma)), (x - mu).T)),
            1.0 * (nD + df) / 2.0,
        )
    )

    return 1.0 * numerator / denominator


def read_chain(chain, burn=1000, thin=10):
    cols = [
        ("magnitude", (-1, 7)),
        ("P", (-1, 7)),
        ("Q__", (-1, 7)),
        ("sigma", (-1, 7, 7)),
        ("NU", (-1, 1)),
        ("c50", (-1, 7)),
    ]
    df = pd.read_csv(chain)
    del df["NU_lowerbound__"]
    out = {}
    for col, shape in cols:
        idcol = [True if x.startswith(col) else False for x in df.columns]
        out[col.replace("__", "")] = df.loc[burn::thin, idcol].values.reshape(shape)
    return out


def plot_pred(trace, idx, step=slice(None, None, 10)):
    import pylab as plt

    P = trace["P"].squeeze()[step, idx]
    Q = trace["Q"].squeeze()[step, idx]
    M = trace["magnitude"].squeeze()[step, idx]
    C = trace["c50"].squeeze()[step, idx]
    sigma = trace["sigma"].squeeze()[step, idx, idx]
    x = np.linspace(1, 100, 100)
    for p, q, m, c in zip(P, Q, M, C):
        y = crf(x, m, p, q, c)
        plt.plot(x, y, "k", alpha=0.05)
    y = crf(x, m.mean(), p.mean(), q.mean(), c.mean())
    plt.plot(x, y, "r", lw=2)
    plt.fill_between(x, y - sigma.mean(), y + sigma.mean(), alpha=0.25, zorder=-1)


def get_trace_for_subject(subject, area):
    import pymc3 as pm

    sa = pd.read_hdf(
        "/home/nwilming/fixed_individual_sample_gp_sltd_remove_overlap.hdf",
        "s_empirical",
    )
    sa = sa.query('subject==%i & area=="%s"' % (subject, area))
    power = pd.pivot_table(sa, values="power", index="sample_id", columns="F")
    contrast = pd.pivot_table(sa, values="contrast", index="sample_id")

    mdl = mv_crf_model(power.values, contrast.values.ravel())
    with mdl:
        savedir = "mvnorm_CRF_S%i_%s_trace" % (subject, area)
        db = pm.backends.Text(savedir)
        trace = pm.sample(4500, tune=1500, cores=4, njobs=4, trace=db)
    import cPickle

    cPickle.dump(
        {"model": mdl, "trace": trace},
        open("mvnorm_CRF_S%i_%s_trace.pkl" % (subject, area), "w"),
    )


def get_decoded_contrast(sa, trace, cc=np.linspace(0, 1, 26)):
    power = pd.pivot_table(sa, values="power", index="sample_id", columns="F")
    contrast = pd.pivot_table(sa, values="contrast", index="sample_id")
    pcd = invert(trace, power.values, contrast.values)
    decoded_contrast = np.dot(pcd.values, pcd.columns.values)
    return pcd, contrast, decoded_contrast


def get_single_decoded_contrast(
    subject, area, trace_path, cc=np.linspace(0, 1, 51), burn=1000, thin=10
):
    import pandas as pd
    import glob

    sa = pd.DataFrame(
        "/home/nwilming/fixed_individual_sample_gp_sltd_remove_overlap.hdf"
    )
    sa = sa.query('subject==%i & area=="%s"' % (subject, area))
    fnames = glob.glob(
        join(base_path, "mvnorm_CRF_S%i_%s_trace" % subject, area, "chain-*.csv")
    )
    out = [read_chain(fname, burn, thin) for fname in fnames]
    keys = list(out[0].keys())
    out = dict((k, np.concatenate([o[k] for o in out])) for k in keys)
    return get_decoded_contrast(sa, out, cc=cc)


def get_all_decoded_contrasts(sa, trace_path, cc=np.linspace(0, 1, 51)):
    from os.path import join

    for (subject, area), data in sa.groupby(["subject", "area"]):
        pass


"""
Plots
"""


def plot_sample_aligned_responses(sa, area="V1-lh"):
    """
    Plot sample aligned responses
    """
    import pylab as plt

    # 1 Plot averaged gp response as a function of contrast and frequency
    dreal = get_average_contrast(sa, by=["subject"], area=area)
    for sub, d in dreal.groupby("subject"):
        plt.plot(d.contrast, d.power, label=sub)
    # 2 Plot predicted gp response as a function of contrast and frequency


"""
CRFs
"""


def crf(x, m, p, q, c):
    k = (m * (x ** (p + q))) / (x ** (q) + c ** (q))
    # ck = (m * (c**(p + q))) / (c**(q) + c**(q))
    return k  # - ck


def crf2(x, m, p, q, c):
    # c = float(c)
    k = (m * (x ** p)) / (x ** (q) + c ** (q))
    ck = (m * (c ** p)) / (c ** (q) + c ** (q))
    return k - ck


def old_crf(x, m, p, q, c):
    c = float(c)
    k = (m * (x ** p)) / (x ** (p + q) + c ** (p + q))
    ck = (m * (c ** p)) / (c ** (p + q) + c ** (p + q))
    return k - ck


def dx_crf(x, m, p, q, c):
    z = m * x ** (p - 1) * (p * (c ** (p + q)) - q * (x ** (p + q)))
    t = (c ** (p + q) + x ** (p + q)) ** 2.0
    return z / t


"""
Precomputing
"""


def foo(x):
    print(x)


def precompute_fits():
    from pymeg import parallel

    subs = list(range(1, 16))
    funcs = ["stld"]
    areas = ["V1-lh", "V2-lh", "V3-lh", "V1-rh", "V2-rh", "V3-rh"]
    from itertools import product

    ids = []
    for sub, area in product(subs, areas):
        F = [40, 45, 50, 55, 60, 65, 70]
        for f in F:
            if not subject_fit.is_cached(sub, f, area):
                ids.append(parallel.pmap(subject_fit, [[sub, f, area]]))
    return ids
