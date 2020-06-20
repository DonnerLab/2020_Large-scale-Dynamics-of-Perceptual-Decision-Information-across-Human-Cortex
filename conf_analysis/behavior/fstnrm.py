import numpy as np
from scipy.stats import norm, gamma
from scipy.stats import t
from scipy.special import gammaln
from numba import jit, float64, int32, int64

#@jit((float64[:], float64[:], int32, float64[:]))
@jit
def NGposterior(xbar, sigma, n, prior):
    '''
    Compute the posterior distribution for n normal samples with mean xbar and std sigma.
    The prior is given by the quadruple (m0, k0, a0, b0) - a Normal-Gamma distribution.
    '''
    mu0, k0, a0, b0 = prior
    mun = (k0*mu0 + n*xbar)/(k0+n)
    kn = k0+n
    an = a0 + (n/2.)
    bn = (b0 + 0.5 *  n*(sigma**2)
                      + (k0*n*((xbar-mu0)**2))
                                  /(2*(k0 + n)))
    return mun, kn, an, bn


#@jit((float64[:], float64[:], int32, float64[:]))
@jit
def Mu_posterior(xbar, sigma, n, prior):
    '''
    Compute the posterior distribution of mu for n normal samples with mean xbar and std sigma.
    The prior is given by the quadruple (m0, k0, a0, b0) - a Normal-Gamma distribution.
    '''
    mun, kn, an, bn = NGposterior(xbar, sigma, n, prior)
    df = 2*an
    loc = mun
    scale = bn/(an*kn)
    return df, loc, scale
