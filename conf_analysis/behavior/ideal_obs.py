import numpy as np
import pylab as plt
from scipy.stats import norm, chi2


def center(bins):
    return np.array([low + (high-low)/2. for low, high in zip(bins[0:-1], bins[1:])])

def pdf(x,s):
    return (x/float(sum(x)))/np.diff(s)[0]

def deriv(f, x, dt=0.0001):
    return (f(x+dt) - f(x-dt))/2.

def sdist(x, sigma, n=10):
    '''
    Calculate PDF of the sampling distribution for standard deviations given the number
    of samples obtained and the standard deviation of the original distribution.

    The sampling distribution is given by:
        (n*S**2)/sigma**2 ~ X2(10)
        g'(S) ~ X2(10)
    where S is the standard deviation. We can therefore not directly evaluate
    the desired pdf. Instead we have to work with this transformed distribution.

    Define:
        g(S) = sqrt(S*sigma**2/n)
        g'(S) = n*S**2/sigma**2
        pdf(g'(S)) = chi2.pdf(g'(S), 10)

    Then the pdf of the transformd variable is given by:
        pfg(S) = chi2.pdf(g'(S), 10)* g'(S)/dy

    See http://math.arizona.edu/~jwatkins/f-transform.pdf
    '''
    f = lambda x: chi2.pdf(x, n)
    nb = n+1
    g_inv = lambda x: (nb*x**2)/(sigma**2)
    dyg_inv = lambda x: (2*nb*x)/(sigma**2)
    y = f(g_inv(x))*dyg_inv(x)
    return y


def mdist(x, mu, sigma, n=10):
    return norm.pdf(x, loc=mu, scale=((sigma**2)/n)**.5)


def make_prior(MU, SIGMA, thresholds, stds=[0.05, 0.1, 0.15]):
    L = lambda x,y: likelihood(MU, SIGMA, x, y)
    if len(thresholds) > 1:
        ts, bins = np.histogram(thresholds, bins=20)
        centers = center(bins)
    else:
        centers = thresholds
        ts = np.array([1])

    ts = ts/float(sum(ts))
    Cl = MU*0
    for w, t in zip(ts, centers):
        l = sum([L(t+.5, s) for s in stds])/len(stds)# + L(t+.5, stds[1]) + L(t+.5, stds[2]))/3.
        Cl += w*l
    return Cl


def likelihood(m,s, mu, sigma, nm=10, ns=10):
    nm = float(nm)
    ns = float(ns)
    m = mdist(m, mu, sigma, nm)
    s = sdist(s, sigma, ns)
    return m*s


def compare_ps(mu, sigma, nm, ns):
    xm = np.linspace(-.5, .5, 500)
    xs = np.linspace(0, 0.3, 250)
    MU, SIGMA = np.meshgrid(xm, xs)
    C = likelihood(MU, SIGMA, mu, sigma, nm, ns)
    s1 = norm.rvs(loc=mu ,scale=sigma, size=(10, 2e6))
    means, sigmas = s1.mean(0), s1.std(0)
    a,b,c = np.histogram2d(means, sigmas, bins=[xm, xs])
    plt.contour(center(xs), center(xm), a/a.sum(), cmap='summer')
    plt.contour(SIGMA, MU, np.exp(C), cmap='winter')


#kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
