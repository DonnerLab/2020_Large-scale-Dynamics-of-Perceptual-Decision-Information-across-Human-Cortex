#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 17:50:18 2017

@author: genis
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import numpy as np
import math
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression


def plotwerr(pivottable, *args, ls="-", label=None, **kwargs):
    import pylab as plt
    import seaborn as sns
    N = pivottable.shape[1]
    x = np.arange(N)    
    mean = pivottable.mean(0)
    std = pivottable.std(0)
    sem = std / (N ** 0.5)
    plt.plot(mean, *args, label=label, **kwargs)
    if "alpha" in kwargs:
        del kwargs["alpha"]
    if "color" in kwargs:
        color = kwargs["color"]
        del kwargs["color"]
        plt.fill_between(
            x,
            mean + sem,
            mean - sem,
            facecolor=color,
            edgecolor="none",
            alpha=0.5,
            **kwargs
        )
    else:
        fill_between(x, mean + sem, mean - sem, edgecolor="none", alpha=0.5, **kwargs)


def get_decision_fluct(data):
    import seaborn as sns
    import pylab as plt
    import matplotlib as mpl
    cvals = np.vstack(data.contrast_probe)
    # Hits:
    with mpl.rc_context(rc={"font.size": 8.0, "lines.linewidth": 1.5}):
        b = np.linspace(0, 1, 21)
        l = .05
        idx = (data.response == -1) & (data.side == -1)
        sns.distplot(
            cvals[idx, 1] + 0 * (data.cbm.values[idx] - 0.1),
            bins=b,
            label="Contrast: Low, Answer: Reference",
            kde=True,
            hist=False,
            kde_kws={"clip": (0, 1), 'bw':l},
            color=list(np.array([29, 119, 0, 100])/255),
        )
        idx = (data.response == 1) & (data.side == -1)
        sns.distplot(
            cvals[idx, 1] + 0 * (data.cbm.values[idx] - 0.1),
            bins=b,
            label="Contrast: Low, Answer: Test",
            kde=True,
            hist=False,
            kde_kws={"clip": (0, 1), 'bw':l},
            color=list(np.array((173, 3, 3, 100))/255),
            
        )
        idx = (data.response == 1) & (data.side == 1)
        sns.distplot(
            cvals[idx, 1] - 0 * (data.cbm.values[idx] + 0.1),
            bins=b,
            label="Contrast: High, Answer: Test",
            kde=True,
            hist=False,
            kde_kws={"clip": (0, 1), 'bw':l},
            color='#2dad03'

        )
        idx = (data.response == -1) & (data.side == 1)
        sns.distplot(
            cvals[idx, 1] - 0 * (data.cbm.values[idx] + 0.1),
            bins=b,
            label="Contrast: High, Answer: Reference",
            kde=True,
            hist=False,
            kde_kws={"clip": (0, 1), 'bw':l},
            color='#ad0303'
        )
        
        plt.xlim([0,1])
        plt.yticks([])
        plt.xticks([0.25, 0.5, .75])
        sns.despine(ax=plt.gca(), left=True)
        plt.legend(frameon=False, loc="upper left", bbox_to_anchor=(-0.1,1))
        plt.xlabel('Contrast')
        yl = plt.ylim()[1]
        plt.ylim([0, yl*2])
    # return pd.concat(kernels)


def get_decision_kernel(df):
    fluct = np.vstack(df.contrast_probe)  #  - (df.cbm * df.side)[:, np.newaxis])
    resp = df.response
    idnan = np.isnan(resp)
    ks = kernel(fluct[~idnan, :], resp.values[~idnan])
    ks = pd.Series(ks[0] - 0.5)
    ks.index.names = ["sample"]
    ks.name = "kernel"
    return ks


def plot_decision_kernel_values(df):
    import pylab as plt
    import seaborn as sns
    side_map = {-1:'reference', 1:'test'}
    resp_map = {-1:'Correct', 1:'Error'}
    colors = {(-1, -1):'#ff7b00', (-1, 1):'#00a5ff', (1, 1):'#ff7b00', (1, -1):'#00a5ff', }
    for i, ((side, resp), d) in enumerate(df.groupby(['side', 'response'])):
        vals = []
        for s, ds in d.groupby('snum'):
            fluct = np.vstack(ds.contrast_probe)   #- (ds.cbm * ds.side)[:, np.newaxis]
            vals.append(fluct.mean(0))
        vals = np.vstack(vals)
        if i >= 2:
            plotwerr(vals, color=colors[(side,resp)])
        else:
            plotwerr(vals, label='Response: %s'%(resp_map[resp]), color=colors[(side,resp)])
    for side, d in df.groupby(['side']):
        fluct = np.vstack(d.contrast_probe)   #- (d.cbm * d.side)[:, np.newaxis]
        plt.plot(fluct.mean(0), 'k--')
    plt.legend(frameon=False)
    plt.xlabel('Sample')
    plt.ylabel('Contrast')
    sns.despine(ax=plt.gca())
    
    

def get_confidence_kernel(df):
    res = {}
    for resp_id, rd in df.groupby("response"):
        fluct = np.vstack(rd.contrast_probe)  #  - (rd.cbm * rd.side)[:, np.newaxis])
        resp = 2 * (rd.confidence - 1.5)
        idnan = np.isnan(resp)
        res[resp_id] = kernel(fluct[~idnan, :], resp.values[~idnan])[0] - 0.5
        if resp_id == -1:
            res[resp_id] = -1 * res[resp_id]
    res = pd.DataFrame(res).stack()
    res.index.names = ["sample", "response"]
    res.name = "kernel"
    return res


def psychometric(x, beta, b):
    return 1.0 / (1 + np.exp(-beta * (x - b)))


def fit_psychometric(coh, PR):
    p, cov = curve_fit(psychometric, coh, PR)
    beta, b = p[:]
    return beta, b, cov


def fit_psychometric_error(coh, d, Nboot):
    """
    Computes sensitivity (beta) and bias (b) and computes its error with bootstrap
    d is a matrix with answers 0 (left) or 1 (right) for each coherence. d[cohereces][trials]
    returns the distributions of beta and b 
    """
    N = len(d[0])
    Nmu = len(d)
    indexs = np.random.randint(0, N, (Nboot, Nmu, N))
    y = np.zeros((Nboot, Nmu))
    beta = np.zeros(Nboot)
    b = np.zeros(Nboot)
    for i in range(Nboot):
        for imu in range(Nmu):
            y[i][imu] = np.mean(d[imu][indexs[i][imu]])
        p, cov = curve_fit(psychometric, coh, y[i])
        beta[i], b[i] = p[:]
    return beta, b


def kernel_logistic(stim, d):
    lr = LogisticRegression()
    lr.fit(stim, d)
    return lr.coef_, lr.intercept_


def kernel_err(stim, d, error=False, Nboot=500):
    """
    Computes the kernel and the standard error with bootstrap.
    inputs:
    stim: 2D-Array of stimulus
    d: 1-D array of decisions (-1,1)

    outputs:
    kernel: Dictionary with kernel and error_kernel
    """
    Nframe = len(stim[0])
    Nstim = len(stim)
    kernel = {"kernel": np.zeros(Nframe), "error": np.zeros(Nframe)}
    d = (-1) * d  # atencio he canviat aixo perque es el reves ###
    area_boot = np.zeros(Nboot)
    PRI = np.zeros(Nboot)
    if not error:
        aux_kernel = np.zeros(Nframe)
        for iframe in range(Nframe):
            fpr, tpr, _ = roc_curve(d, stim[:, iframe])
            aux_kernel[iframe] = auc(tpr, fpr)
        kernel["kernel"] = aux_kernel
    else:
        #        if Nboot is not None:
        #            Nboot=len(stim)

        aux_kernel = np.zeros((Nframe, Nboot))
        indexs = np.random.randint(0, Nstim, (Nboot, Nstim))
        #        print indexs[0]
        #        print d[indexs[0]]

        for iboot in range(Nboot):
            if iboot % 100 == 0:
                print(iboot)
            for iframe in range(Nframe):
                fpr, tpr, _ = roc_curve(d[indexs[iboot]], stim[indexs[iboot], iframe])
                aux_kernel[iframe][iboot] = auc(tpr, fpr)

            area_boot[iboot] = total_area_kernel_PInormalize(aux_kernel.T[iboot])

            #            area_boot[iboot]=np.sum(aux_kernel.T[iboot])
            PRI[iboot] = primacy_recency_ratio(aux_kernel.T[iboot])

        for iframe in range(Nframe):
            kernel["kernel"][iframe] = np.mean(aux_kernel[iframe])
            kernel["error"][iframe] = np.std(aux_kernel[iframe])
        area_total = {}
        area_total["area"] = np.mean(area_boot)
        area_total["err"] = np.std(area_boot)

        PRI_total = {}
        PRI_total["pri"] = np.mean(PRI)
        PRI_total["err"] = np.std(PRI)

    return kernel, area_total, PRI_total


#    return kernel,area_total,PRI_total


def kernel(stim, d):
    """
    Computes the kernel and the standard error with bootstrap.
    inputs:
    stim: 2D-Array of stimulus
    d: 1-D array of decisions (-1,1)

    outputs:
    kernel: Dictionary with kernel and error_kernel
    """
    Nframe = len(stim[0])
    Nstim = len(stim)
    kernel = np.zeros(Nframe)
    d = (-1) * d
    aux_kernel = np.zeros(Nframe)
    for iframe in range(Nframe):
        fpr, tpr, _ = roc_curve(d, stim[:, iframe])
        aux_kernel[iframe] = auc(tpr, fpr)
    kernel = aux_kernel
    return kernel, None


def stim2frames(stim, Nframes):
    if len(stim[0]) % Nframes != 0:
        print("wrong Nframes")
    else:
        window = len(stim[0]) / Nframes
        stim_f = np.zeros((len(stim), Nframes))
        for istim in range(len(stim)):
            for iframe in range(Nframes):
                stim_f[istim][iframe] = np.mean(
                    stim[istim][iframe * window : (iframe + 1) * window]
                )

    return stim_f


def rate2rateframes(stim, Nframes):
    if len(stim) % Nframes != 0:
        print("wrong Nframes")
    else:
        window = len(stim) / Nframes
        stim_f = np.zeros(Nframes)
        for iframe in range(Nframes):
            stim_f[iframe] = np.mean(stim[iframe * window : (iframe + 1) * window])

    return stim_f


def pd_kernel_slope(df):
    from scipy.stats import linregress

    kernel = df.groupby("sample").mean().values
    ps = linregress(np.arange(len(kernel)), kernel)
    return ps[0]


def pd_primacy_recency_ratio(df):
    kernel = df.groupby("sample").mean().values
    return primacy_recency_ratio(kernel + 0.5)


def pd_total_area_kernel(df):
    kernel = df.groupby("sample").mean().values
    return total_area_kernel(kernel + 0.5)


def primacy_recency_ratio(kernel):
    """
    Compute the primacy vs recency ratio as:
    PRR=integral( kernel*f(t))
    with f(t)=1-a*t with a such as f(T)=-1 T stimulus duration
    positive primacy
    zero flat
    negative primacy
    """
    aux = np.linspace(1, -1, len(kernel))
    kernel = kernel - 0.5
    aux_kernel = (kernel) / (np.sum(kernel))
    return np.sum(aux_kernel * aux)


def primacy_recency_ratio_half_half(kernel):
    """
    Compute the PRR with P=sum(kernel[0:T/2]) and R=sum(kernel[T/2:T])
    """
    return sum(kernel[0 : len(kernel) / 2]) / sum(kernel[len(kernel) / 2 : -1])


def total_area_kernel(kernel, T=1):
    """
    Compute the PRR with P=sum(kernel[0:T/2]) and R=sum(kernel[T/2:T])
    """

    return sum(kernel - 0.5) / len(kernel) * T


def total_area_kernel_PInormalize(kernel):
    """
    Compute the PRR with P=sum(kernel[0:T/2]) and R=sum(kernel[T/2:T])
    """
    nframes = len(kernel)
    area_pi = (
        nframes * (0.5 + 2 / np.pi * np.arctan(1 / np.sqrt(2 * nframes - 1)))
        - 0.5 * nframes
    )

    return np.sum(kernel - 0.5) / area_pi


def make_control_stim(mu, sigma, T, N):
    """
    it returns stimulus with control mu and sigma
    """
    if N == 1:
        stim = np.random.randn(T)
        m = np.mean(stim)
        s = np.std(stim)
        stim = mu + ((stim - m) / s) * sigma
    else:
        stim = np.random.randn(N, T)
        for itrial in range(N):
            m = np.mean(stim[itrial])
            s = np.std(stim[itrial])
            stim[itrial] = mu + ((stim[itrial] - m) / s) * sigma
    return stim


def mean_and_error(vector):
    """
    mean and error according to binomial distribution
    """

    m = np.mean(vector)
    z = 1
    # print len(vector)
    return np.mean(vector), z * np.sqrt(m * (1 - m) / len(vector))


def stationary(x, mu, a, b, D, N):
    return N * np.exp(-potential(x, mu, a, b) / D)


def potential(x, mu, a, b):
    return mu * x + a * x ** 2 + b * x ** 4


def potential_prima(x, mu, a, b):
    return mu + 2.0 * a * x + 4.0 * b * x ** 3


def potential_prima2(x, a, b):
    return 2.0 * a + 12 * b * x ** 2


def func(x, a, b):
    return 1.0 / (1.0 + np.exp(-(a * x - b)))


def trap(a1, a2, mu, a, b, D, n):
    h = (a2 - a1) / (n - 1)
    value = 0.5 * (
        np.exp(potential(a1, mu, a, b) / D) + np.exp(potential(a2, mu, a, b) / D)
    )
    for i in range(n):
        value += np.exp(potential(a1 + h * (i - 1), mu, a, b) / D)
    value *= h
    return value


def Performance_sigma(sigmas_theory, coef, t=100.0, sigma0=0.0):

    Ntrap = 100

    x0 = 0.0
    mu = coef[0]
    a = coef[1]
    b = coef[2]

    PR0 = np.zeros((len(sigmas_theory)))
    PRs = np.zeros((len(sigmas_theory)))
    PRR = np.zeros((len(sigmas_theory)))
    PRL = np.zeros((len(sigmas_theory)))
    PR = np.zeros((len(sigmas_theory)))

    for isigma in range(len(sigmas_theory)):
        sigma = np.sqrt(sigma0 ** 2 + sigmas_theory[isigma] ** 2)
        D = 1.0 * sigma ** 2 / 2.0

        p = np.zeros(4)
        p[0] = 4.0 * b
        p[2] = 2.0 * a
        p[3] = mu
        z = np.roots(p)

        za = np.min(z)
        zc = np.max(z)
        zb = np.median(z)
        PR0[isigma] = trap(za, x0, mu, a, b, D, Ntrap) / trap(
            za, zc, mu, a, b, D, Ntrap
        )
        Delta_ab = potential(zb, mu, a, b) - potential(za, mu, a, b)
        Delta_cb = potential(zb, mu, a, b) - potential(zc, mu, a, b)
        # Rate from ac and ca
        aux_kac = np.sqrt(
            np.abs(potential_prima2(zb, a, b) * potential_prima2(za, a, b))
        ) / (2.0 * np.pi)
        kac = aux_kac * np.exp(-Delta_ab / D)

        aux_kca = np.sqrt(
            np.abs(potential_prima2(zb, a, b) * potential_prima2(zc, a, b))
        ) / (2.0 * np.pi)
        kca = aux_kca * np.exp(-Delta_cb / D)

        PRs[isigma] = kac / (kac + kca)
        PRR[isigma] = PRs[isigma] + np.exp(-(kca + kac) * t) * (1 - PRs[isigma])
        PRL[isigma] = PRs[isigma] * (1 - np.exp(-(kca + kac) * t))
        PR[isigma] = PRR[isigma] * PR0[isigma] + PRL[isigma] * (1 - PR0[isigma])
    return PR, PRL, PRR, PR0, PRs


def Performance_t_fit(t, a, b, sigma0):

    Ntrap = 100

    x0 = 0.0
    mu = -0.1
    sigma = 0.15 * np.sqrt(2)
    sigma = np.sqrt(sigma0 ** 2 + sigma ** 2)
    D = 1.0 * sigma ** 2 / 2.0

    p = np.zeros(4)
    p[0] = 4.0 * b
    p[2] = 2.0 * a
    p[3] = mu
    z = np.roots(p)

    za = np.min(z)
    zc = np.max(z)
    zb = np.median(z)
    PR0 = trap(za, x0, mu, a, b, D, Ntrap) / trap(za, zc, mu, a, b, D, Ntrap)
    Delta_ab = potential(zb, mu, a, b) - potential(za, mu, a, b)
    Delta_cb = potential(zb, mu, a, b) - potential(zc, mu, a, b)
    # Rate from ac and ca
    aux_kac = np.sqrt(
        np.abs(potential_prima2(zb, a, b) * potential_prima2(za, a, b))
    ) / (2.0 * np.pi)
    kac = aux_kac * np.exp(-Delta_ab / D)

    aux_kca = np.sqrt(
        np.abs(potential_prima2(zb, a, b) * potential_prima2(zc, a, b))
    ) / (2.0 * np.pi)
    kca = aux_kca * np.exp(-Delta_cb / D)

    PRs = kac / (kac + kca)
    PRR = PRs + np.exp(-(kca + kac) * t) * (1 - PRs)
    PRL = PRs * (1 - np.exp(-(kca + kac) * t))
    PR = PRR * PR0 + PRL * (1 - PR0)
    return PR
