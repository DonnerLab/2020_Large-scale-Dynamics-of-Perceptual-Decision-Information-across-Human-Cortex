import numpy as np
from conf_analysis.behavior import normative as nrm

from scipy.stats import bernoulli

def get_prior(data, N_samples):
    exprior = np.array([0, .1, .1, .1])
    cvals = np.vstack(data.contrast_probe)
    prior = nrm.NGposterior(cvals.mean(), cvals.std(), N_samples, exprior)
    return prior


def decide_trial(lpr, dcut=2.5, ccut=4, alpha=None, offset=None):
    '''
    Decide a set of trials based on decision threshold and confidence threshold
    (first passage is absorbing)
    '''
    responses = lpr[:,0]*0
    first_passage_times = np.zeros((lpr.shape[0], ))+9
    for k in range(len(lpr)):
        l = lpr[k, :]
        for i in range(len(l)):
            if l[i]>dcut:
                responses[k] = 1
                first_passage_times[k] = i
                break
            elif l[i]<-dcut:
                responses[k] = -1
                first_passage_times[k] = i
                break
            responses[k] = np.sign(l[i])

    if alpha is None:
        confidence = 1+ ((abs(lpr)>ccut).sum(1)>1)
    else:
        p = (1+exp(-alpha*(first_passage_times + offset - np.median(first_passage_times))))**-1
        confidence = bernoulli.rvs(p)+1
    return responses, confidence, first_passage_times


from numba import jit

@jit
def decide_trial_prob(lpr, dcut=2.5, alpha=None, offset=None):
    '''
    Decide a set of trials based on decision threshold (first passage is absorbing).
    Confidence is determined by the first passage time. Probability of a high confidence
    judgment is modelled with a sigmoidal function (modelled by slope=alpha and
    threshold=offset)
    '''
    responses = lpr[:,0]*0
    first_passage_times = np.zeros((lpr.shape[0], ))+9
    p_resp = lpr[:, 0]*0 + 0.5
    for k in range(len(lpr)):
        l = lpr[k, :]
        for i in range(len(l)):
            if l[i]>dcut:
                responses[k] = 1
                first_passage_times[k] = i
                p_resp[k] = 0.9
                break
            elif l[i]<-dcut:
                responses[k] = -1
                first_passage_times[k] = i
                p_resp[k] = 0.1
                break
            responses[k] = np.sign(l[i])

    p = ((1+np.exp(-alpha*(first_passage_times + offset - np.median(first_passage_times))))**-1)
    confidence = bernoulli.rvs(p)+1
    return responses, confidence, first_passage_times, p, p_resp.astype(float)


#@jit
def determine_first_passage_thresholds(dz, field='LPR',
                      alphas=np.linspace(-1, 1, 13),
                      offset=np.linspace(-3.5, 3.5, 11),
                      dcut=np.linspace(0, 5., 12),
                      bias=0):

    lprvals = np.vstack(dz.loc[:, field].values)+bias
    Al, Off, Dc = np.meshgrid(alphas, offset, dcut)
    scores = (Al*np.nan).ravel()

    for i, (alpha, off, dcut) in enumerate(zip(Al.ravel(), Off.ravel(), Dc.ravel())):
        resp, conf, fp, p, p_r = decide_trial_prob(lprvals, dcut=dcut, alpha=alpha, offset=off)
        x = (dz.confidence.values-1)
        x_r = (dz.response.values==1).astype(float)
        score = np.sum(-np.log((p**x) * (1-p)**(1-x)))
        score += np.sum(-np.log((p_r**x_r) * (1-p_r)**(1-x_r)))
        scores[i] = score

    idt = np.argmin(scores)
    return (Al.ravel()[idt], Off.ravel()[idt], Dc.ravel()[idt],
            np.array(scores).reshape(Al.shape), idt,
            Al, Off, Dc)



def determine_ccut(dz, dcut, field='LPR', ccuts=np.linspace(0, 20, 31)):
    '''
    Determine confidence threshold by maximizing the number of correctly predicted
    confidence judgments.
    '''
    lprvals = np.vstack(dz.loc[:, field].values)
    scores = []
    for ccut in ccuts:
        resp, conf,_ = decide_trial(lprvals, dcut=dcut, ccut=ccut)
        id2 = ~np.isnan(dz.confidence.values) & (dz.confidence.values==2)
        id1 = ~np.isnan(dz.confidence.values) & (dz.confidence.values==1)
        score = (np.mean(conf[id2]==2) + np.mean(conf[id1]==1))/2.
        scores.append(score)
    idt = np.argmax(scores)
    return ccuts[idt], scores, idt


def determine_dcut(dz, field='LPR', dcuts=np.linspace(0, 10, 31)):
    '''
    Determine decision cut off by maximizing the number of correctly predicted
    respons.
    '''
    lprvals = np.vstack(dz.loc[:, field].values)
    proportions = []
    for dcut in dcuts:
        resp, conf,_ = decide_trial(lprvals, dcut=dcut, ccut=10)
        proportions.append(np.mean(resp==dz.response.values))

    proportions = np.array(proportions)
    idt = np.argmax(abs(proportions-(dz.response).mean()))
    return dcuts[idt], proportions[idt]
