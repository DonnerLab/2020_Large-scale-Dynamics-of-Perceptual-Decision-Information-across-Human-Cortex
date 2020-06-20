'''
Implements a normative model that infers mean of a sample in the face of unknown
mean and variance of the generative distribution.
'''
import numpy as np
from scipy.stats import norm, gamma
from scipy.stats import t
from scipy.special import gammaln
from conf_analysis.behavior import fstnrm as fast

def NG(mu, precision, mu0, kappa0, alpha0, beta0):
    '''
    PDF of a Normal-Gamma distribution.
    '''
    return norm.pdf(mu, mu0, 1./(precision*kappa0)) * gamma.pdf(precision, alpha0, scale=1./beta0)


def NGposterior(xbar, sigma, n, prior):
    '''
    Compute the posterior distribution for n normal samples with mean xbar and std sigma.
    The prior is given by the quadruple prior (m0, k0, a0, b0) - a Normal-Gamma distribution.
    '''
    mu0, k0, a0, b0 = prior
    mun = (k0*mu0 + n*xbar)/(k0+n)
    kn = k0+n
    an = a0 + (n/2.)
    bn = (b0 + 0.5 *  n*(sigma**2)
                      + (k0*n*((xbar-mu0)**2))
                                  /(2*(k0 + n)))
    return mun, kn, an, bn


def Mu_posterior(xbar, sigma, n, prior):
    '''
    Compute the posterior distribution of mu for n normal samples with mean xbar and std sigma.
    The prior is given by the quadruple prior (m0, k0, a0, b0) - a Normal-Gamma distribution.
    '''
    mun, kn, an, bn = NGposterior(xbar, sigma, n, prior)
    df = 2*an
    loc = mun
    scale = bn/(an*kn)
    return df, loc, scale


def Sigma_posterior(xbar, sigma, n, prior):
    '''
    Compute the posterior distribution of sigma for n normal samples with mean xbar and std sigma.
    The prior is given by the quadruple prior (m0, k0, a0, b0) - a Normal-Gamma distribution.
    '''
    mun, kn, an, bn = NGposterior(xbar, sigma, n, prior)
    return an, 1./bn


def plarger(mean, std, n, prior):
    '''
    Calculate the probability that a sample has a mean value larger than 0. This
    works by marginalizing out the standard deviation.
    '''
    df, loc, scale = fast.Mu_posterior(mean.ravel(), std.ravel(), n, prior)
    return 1.-t(df,loc,scale).cdf(0).reshape(mean.shape)


def psmaller(mean, std, n, prior):
    '''
    Calculate the probability that a sample has a mean value smaller than 0:
        psmaller(m,s,n, prior) = 1-plarger(mean, std, n, prior)
    '''
    df, loc, scale = fast.Mu_posterior(mean.ravel(), std.ravel(), n, prior)
    return t(df,loc,scale).cdf(0).reshape(mean.shape)


def logLPR(mean, std, prior, N=10):
    '''
    This gives the log posterior ration. Since probabilities can
    easily become 0 or 1 I make sure to clip them to 0+eps and 1-eps.
    '''
    mean = np.asarray(mean)
    std = np.asarray(std)
    pL = plarger(mean, std, N, prior)
    pL = np.maximum(np.minimum(pL, 1-np.finfo(float).eps), 0+np.finfo(float).eps)
    #pS = psmaller(mean, std, 10, prior)
    #pS = np.maximum(np.minimum(pS, 1-np.finfo(float).eps), 0+np.finfo(float).eps)
    return np.log(pL) - np.log(1-pL)


def cumulative_LLPR(samples, prior):
    '''
    Compute the log posterior along the columns in matrix samples.
    '''
    results = samples*0
    for k in range(1, samples.shape[1]+1):
        mean, std = samples[:, :k].mean(1), samples[:, :k].std(1)
        results[:, k-1] = logLPR(mean, std, prior, N=k)
    return results


def get_samples(N=10, thresh=0.25, symmetric=True, vars=[0.05, 0.1, 0.15]):
    '''
    Generate a set of samples in analogy to my experiment.
    '''
    if not symmetric:
        return np.concatenate(
              [np.random.randn(N, 10)*v + thresh for v in vars])
    else:
        return np.concatenate(
              [np.random.randn(N, 10)*v + thresh for v in vars]
             +[np.random.randn(N, 10)*v - thresh for v in vars])


class IdealObserver(object):
    '''
    Represents an ideal observer with a fixed prior.
    '''

    def __init__(self, bias, conf_threshold, prior=np.array([0., .01, .01, .01])):
        self.bias=float(bias)
        self.conf_threshold = float(conf_threshold)
        self.prior = prior

    def __str__(self):
        s = 'Bias: %2.1f, conf_threshold: %2.1f, prior:'%(self.bias, self.conf_threshold)
        for i in self.prior:
            s +=  '%2.4f  '%i
        return s

    def __call__(self, mean, sigma, index=False):
        '''
        Return a decision (-2, -1, 1, 2) for sample with mu=mean and sigma=sigma.
        '''

        LPR = logLPR(mean, sigma, self.prior)-self.bias
        confidence = 1 + (abs(LPR) >  self.conf_threshold)
        choice = np.sign(LPR)
        choice[choice==0]+=1
        if not index:
            return choice*confidence
        else:
            index = choice*confidence+2
            index[index>1] -= 1
            return index

    def p(self, mean, sigma):
        '''
        Return multinomial probability vector for samples.
        '''
        decision = self(mean, sigma, index=True)
        ps = np.ones((4, len(decision)))*0 + 0.01
        for i, idx in enumerate(decision):
            ps[idx, i] = 1
        return ps/ps.sum(0)


from scipy import optimize

class NrmModel(IdealObserver):

    def __init__(self, bias=0, conf_threshold=np.nan, mu0=0., kappa0=1., alpha0=1., beta0=1.):
        prior = np.array([mu0, kappa0, alpha0, abs(beta0)])
        super(self.__class__, self).__init__(bias, conf_threshold, prior=prior)

    def fit(self, X,y, opt_fit=False, libcma=False, **params):
        if opt_fit:
            def err_function(params):
                mdl = NrmModel(bias=params[0], conf_threshold=params[1], mu0=params[2],
                               kappa0=params[3], alpha0=params[4], beta0=params[5])
                return -mdl.score(X,y)
            start = np.array([self.bias, self.conf_threshold] + list(self.prior))
            if not libcma:
                import cma
                x = cma.fmin(err_function, start, 1.5, restarts=5, options={
                                'verbose':-9})[0]
            else:
                import lcmaes
                import lcmaes_interface as lci
                print(('Start:', err_function(start)))
                ffunc = lci.to_fitfunc(err_function)
                lbounds = [-np.inf, 0, -np.inf] + 3*[0]

                fopt = lci.to_params(list(start), 1.5,
                    str_algo=b'aipop',
                    lbounds=lbounds,
                    restarts=5,
                    )

                res = lci.pcmaes(ffunc, fopt)
                bcand = res.best_candidate()
                print(('End:', bcand.get_fvalue()))
                x = lcmaes.get_candidate_x(bcand)
            return self.set_params(bias=x[0], conf_threshold=x[1], mu0=x[2], kappa0=x[3], alpha0=x[4], beta0=x[5])
        return self

    def score(self, X, y):
        '''
        Approximates the likelihood of data given a set of parameters.
        This is actually the formula for a multinomial distribution with all
        the constants taken out and considering that each trial only has one
        choice.
        '''
        X = X.T
        p = self(X[0].ravel(), X[1].ravel())
        idx = ((y==p)/2.)+0.25
        return sum(np.log(idx))


    def predict(self, X):
        X = X.T
        return self(X[0], X[1])

    def get_params(self, deep=True):
        return {'bias': self.bias, 'conf_threshold':self.conf_threshold,
                'mu0':self.prior[0], 'kappa0':self.prior[1],
                'alpha0':self.prior[2], 'beta0':self.prior[3]}

    def set_params(self, **ps):
        self.bias = ps['bias']
        self.conf_threshold = ps['conf_threshold']
        self.prior = np.array([ps['mu0'], ps['kappa0'], ps['alpha0'], ps['beta0']])
        return self

def fit_one_sub(data):
    import pickle
    fitted_obs, bias, cutoff, prior, x, _, _ = fit_single(data)
    pickle.dump(
        {'bias':bias, 'conf_threshold':cutoff, 'prior': prior, 'snum':snum},
        open('nrm_fit_parameters_s%i.pickle'%snum)
    )


def multinomial(xs, ps):
   """
   Return the probability to draw counts xs from multinomial distribution with
   probabilities ps.

   Returns:
       probability: float
   """
   xs, ps = np.array(xs), np.array(ps)
   n = sum(xs)

   result = gammaln(n+1) - sum(gammaln(xs+1)) + sum(xs * np.log(ps))
   return -result


def vec_multinomial(xs, ps):
   """
   Return the probability to draw counts xs from multinomial distribution with
   probabilities ps.

   Returns:
       probability: float
   """
   xs, ps = np.array(xs), np.array(ps)
   n = sum(xs, 0)

   result = gammaln(n+1) - sum(gammaln(xs+1), 0) + sum(xs * np.log(ps), 0)
   return -result


def err_fct(parameters, true, data):
    '''
    Computes the likelihood of data given a set of parameters.
    '''
    prior = parameters[2:]
    obs = IdealObserver(bias=parameters[0], conf_threshold=np.log(parameters[1]), prior=prior)

    answers = np.ones((4, len(true)))*0
    for i, idx in enumerate(true):
        answers[idx, i] = 1

    ps = obs.p(data[0], data[1])
    val = vec_multinomial(answers, ps).sum()
    return val

def opt_err_fct(parameters, true, data):
    '''
    Approximates the likelihood of data given a set of parameters.
    '''
    prior = parameters[2:]
    prior[1:] = abs(prior[1:])
    obs = IdealObserver(bias=parameters[0], conf_threshold=np.log(parameters[1]), prior=prior)
    p = obs(data[0], data[1])
    idx = ((true==p)/2.)+0.25
    return -sum(np.log(idx))

def p2s(precision):
    '''
    Convert precision to standard deviation.
    '''
    return (1./precision)**.5


def s2p(sigma):
    '''
    Convert standard deviation to precision.
    '''
    return 1./(sigma**2)
