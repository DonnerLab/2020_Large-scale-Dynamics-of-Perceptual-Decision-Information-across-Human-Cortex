import numpy as np
from scipy.stats import linregress
from scipy.optimize import least_squares
from pylab import *

from conf_analysis.behavior import kernels
def crf(contrast, bias=0):
    r =  2*(1/(1+np.exp(-(contrast+bias))))-1
    return r


def hcrf(C, p=2.3, q= -8.68016592e-06, a=.0000575, sigma=1):
    #p = 0.33
    #q = 1.55
    #a = 1.48
    #sigma = 1.64
    return a*((C**(p+q))/(C**q + sigma**p))


def err_pqa(x0, crf):
    p, q, a = x0    
    s = 1
    return hcrf(crf.pc, p, q, a, s)-(crf.loc[:, 0]-crf.loc[:, 0].min())


def err_pq(x0, crf):
    p, q = x0
    a = 0.00001
    s = 1
    return hcrf(crf.pc, p, q, a, s)-(crf.loc[:, 0]-crf.loc[:, 0].min())

def err_a(x0, p, q, crf):    
    a = x0
    s=1
    return hcrf(crf.pc, p, q, a, s)-(crf.loc[:, 0]-crf.loc[:, 0].min())


def fit_hcrf2(crf):
    P, Q = np.meshgrid(np.linspace(0.1, 5, 41), np.linspace(0.1, 5, 41))
    A = np.linspace(0.0001, 1, 100)
    cost = []
    
    for p, q in zip(P.ravel(), Q.ravel()):
        for a in A:
            pass
        a.append(params.x[0])
        cost.append(params.cost)
    idx = np.argmin(cost)
    return P.ravel()[idx], Q.ravel()[idx], a[idx], cost


def fit_hcrf(crf):    
    #print('Fitting p,q')
    params = least_squares(lambda x: err_pq(x, crf), 
        [1, 1], max_nfev=1000, ftol=1e-10, method='dogbox')
    p, q = params.x
    #print(p, q)
    #print('Fitting a')
    params = least_squares(lambda x: err_a(x, p, q, crf), 
        [1,], max_nfev=1000, ftol=1e-10, method='dogbox')
    a = params.x[0]
    #print('Fitting p, q, a together')
    params = least_squares(lambda x: err_pqa(x, crf), 
        [p, q, a], max_nfev=1000, ftol=1e-10, method='dogbox')
    return p, q, a#params.x

def example(crf):
        c0 = np.linspace(0, 50, 100)
        c1 = np.linspace(50, 100, 100)

        stim = np.concatenate([-1+0*c0, 1+0*c1])
        contrast = np.concatenate([c0, c1])

        #figure(figsize=(10, 5))
        #subplot(1,2,1)
        p, q, a = fit_hcrf(crf)
        d = np.concatenate([hcrf(c0,p=p, q=q, a=a), hcrf(c1, p=p, q=q, a=a)])
        #d = d/d.max()
        s, i, _, _, _ = linregress(contrast,  d)

        plot(contrast, (stim+1)/2, 'k', label='Choice', alpha=0.5, zorder=-1)
        plot(contrast, d, label='Neural response')
        plot(contrast, contrast*s+i, label='Linear fit', alpha=0.5)
        plot(contrast, d-(contrast*s+i), label='Residuals', alpha=0.5)
        plot(crf.pc, crf.loc[:, 0]-crf.loc[:, 0].min(), 'r', label='Empirical CRF')
        k = kernels.kernel((d-(contrast*s+i))[:, None], stim[:, None])[0]-0.5
        text(1, 0.6, r'$AUC-0.5:%0.2f$'%k)
        legend()
        xlabel('Contrast')
        ylabel('a.u.')
        #title('Bias=0')

        """
        subplot(1,2,2)

        d = np.concatenate([crf(c0, bias=3), crf(c1, bias=3)])
        s, i, _, _, _ = linregress(contrast,  d)

        plot(contrast, stim, 'k', label='Choice')
        plot(contrast, d, label='Neural respons')
        plot(contrast, contrast*s+i, label='Linear fit')
        plot(contrast, d-contrast*s+i, label='Residuals')
        k = kernels.kernel((d-(contrast*s+i))[:, None], stim[:, None])[0]-0.5
        text(1, -1, r'$AUC-0.5:%0.2f$'%k)
        #legend()
        xlabel('Contrast')
        ylabel('a.u.')
        
        title('Bias=3')
        """

def emp_crf(subject):
    X, tpc, freq, meta = ck.build_design_matrix(subject, 'vfcPrimary', 0.19, 'Averaged', freq_bands=[45, 65], ogl=False)
    contrast = np.stack(meta.contrast_probe)
    con = pd.DataFrame(contrast).stack().reset_index()
    con.columns = ['trial', 'sample', 'contrast']
    power = pd.DataFrame(X).stack().reset_index()
    power.columns = ['trial', 'sample', 'power']
    power.loc[:, 'contrast'] = con.contrast
    power.loc[:, 'subject'] = subject
    return power.groupby([pd.cut(power.contrast, np.linspace(0, 1, 9)), 'sample']).mean()



    
