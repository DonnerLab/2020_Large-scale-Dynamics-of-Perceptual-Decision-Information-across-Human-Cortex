'''
Implement variance misperception model from Zylberberg et al. 2014

This module works with a pandas data frame that has the following fields:
    - conf_0index: 0 for low confidence, 1 for high confidence
    - mu: mean contrast in a trial
    - correct: 0 if error, 1 if correct choice

'''
from pylab import *
from scipy.stats import norm
from scipy.optimize import minimize_scalar, minimize
import numpy as np
import pandas as pd


# Fit mean variance level:
def fit_sigma(correct, mu):
    '''
    Fit std of internal 'noise distribution' Zylberberg et al. 2014 style:
    "
    To derive the standard error of the internal evidence distribution (rAcc),
    we assumed that the likelihood function p(x| mu_j, sigma_acc_j) was normally
    distributed. We estimated sigma_acc_j for the different levels of jitter by
    setting the mean (mu) to the orientation determined by the staircase procedure.
    We estimated sigma_acc_j by matching the proportion of incorrect responses per
    participant and variability class (Green & Sweets, 1966), assuming that
    subjects selected option CW if x > 0 and option CCW otherwise (i.e., solved
    p_err_j = Int_-inf^0 N(x| mu_j, sigma_acc_j), where p_err_j is the
    proportion of errors made at jitter level j.
    "

    Input:
        correct : Vector where each index encodes a trial and each entry encodes
            whether decision was correct.
        mu : Mean contrast (or generative mean) of a trial, 0 should be the optimal
            decsion boundary.
    Output:
        std of 'internal distribution'
    '''
    errors = ~correct
    prediction = lambda x: norm.cdf(0, mean(mu), x)
    errf = lambda x: sum((errors-prediction(x))**2)
    return minimize_scalar(errf, method='Bounded', bounds=[0, 100000])


def logPR(w, trial_contrast, trial_sigma, threshold, sigmas):
    sigma = w*trial_sigma + (1.-w)*mean(sigmas)
    return (
            log(norm.pdf(trial_contrast,  threshold, sigma))
          - log(norm.pdf(trial_contrast, -threshold, sigma))
           )

def predict_confidence(w, c, trial_contrast, trial_sigma, threshold, sigma_list):
    '''
    '''
    lpr = logPR(w, trial_contrast, trial_sigma, threshold, sigma_list)
    return abs(lpr)>c

def conf_vs_noise(data):
    return (data.groupby('noise_sigma')
         .conf_0index.mean().reset_index()
         .pivot_table(index='noise_sigma'))


def predict_conf_vs_noise(data, w, c):
    p_high_conf = fit_internal_sigma(data)
    sigma_int = dict((k, v) for k, v in zip(p_high_conf.index.values, p_high_conf.values))
    sigmas = array([sigma_int[n] for n in data.noise_sigma]).ravel()
    return predict_confidence(w, c, data.mc, sigmas, mean(data.contrast), list(sigma_int.values()))


def predicted_conf_vs_noise(data, w, c):
    p_high_conf = fit_internal_sigma(data)
    sigma_int = dict((k, v) for k, v in zip(p_high_conf.index.values, p_high_conf.values))
    sigmas = array([sigma_int[n] for n in data.noise_sigma]).ravel()
    confidence = predict_confidence(w, c, data.mc, sigmas, mean(data.contrast), list(sigma_int.values()))
    predicted = array([mean(confidence[data.noise_sigma.values==ns]) for ns in sort(list(sigma_int.keys()))])
    return predicted


def fit_internal_sigma(data):
    return (data.groupby('noise_sigma')
        .apply(lambda x: fit_sigma(x.correct.astype(bool), x.mc).x)
        .reset_index().pivot_table(index='noise_sigma'))


def explicit(confidence, choice, trial_contrast, threshold,
            trial_sigma, side, p_e=None):
    '''
    Fit variance misperception model.

    '''
    w, s = meshgrid(linspace(0., 1., 101), linspace(0, 2, 51))

    mcadjusted = trial_contrast.values.copy()
    mcadjusted[~side.values.astype(bool)] *= -1

    if p_e is None:
        p_e = fit_internal_sigma(
            pd.DataFrame({'correct':choice,
                          'mc':mcadjusted,
                          'noise_sigma':trial_sigma}))

    sigma_int = dict((k, v) for k, v in zip(p_e.index.values, p_e.values))
    sigmas = array([sigma_int[n] for n in trial_sigma]).ravel()

    target = conf_vs_noise(
                    pd.DataFrame({'conf_0index':confidence,
                                  'noise_sigma':trial_sigma})).values.ravel()

    def object_func(w, c):
        confidence = predict_confidence(w, c, trial_contrast, sigmas,
                                        threshold, list(sigma_int.values()))
        predicted = array([mean(confidence[trial_sigma.values==ns]) for ns in p_e.index.values])
        return sum((target-predicted)**2)

    res = []
    for wb, sb in zip(w.ravel(), s.ravel()):
        res.append(object_func(wb, sb))
    res = array(res).reshape(w.shape)
    m = argmin(res)
    conf = predict_confidence(w.ravel()[m], s.ravel()[m], trial_contrast, sigmas,
                                    threshold, list(sigma_int.values()))
    predicted = array([mean(conf[trial_sigma.values==ns]) for ns in p_e.index.values])
    return w, s, res, target, predicted, p_e


def grid2D(data):
    '''
    Fit variance misperception model with a 2D grid search.
    '''

    w, s = meshgrid(linspace(0., 1., 51), linspace(0., 5, 51))

    p_e = fit_internal_sigma(data)
    sigma_int = dict((k, v) for k, v in zip(p_e.index.values, p_e.values))
    sigmas = array([sigma_int[n] for n in data.noise_sigma]).ravel()
    target = conf_vs_noise(data).values.ravel()

    def object_func(w, c):
        confidence = predict_confidence(w, c, data.mc-.5, sigmas, mean(data.contrast), list(sigma_int.values()))
        predicted = array([mean(confidence[data.noise_sigma.values==ns]) for ns in p_e.index.values])
        return sum((target-predicted)**2)
    res = []
    for wb, sb in zip(w.ravel(), s.ravel()):
        res.append(object_func(wb, sb))
    res = array(res).reshape(w.shape)
    m = argmin(res)
    #return w.ravel()[m], s.ravel()[m]
    return w, s,
