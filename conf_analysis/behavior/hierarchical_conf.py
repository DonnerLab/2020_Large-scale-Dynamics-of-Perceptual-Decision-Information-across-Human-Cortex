import pandas as pd
import numpy as np
import pystan
import patsy
import seaborn as sns
from scipy.special import expit
from scipy.stats import binom
import pylab as plt
import pickle

def qcut(x, bins, name):
    labels = np.around([(low + (high-low)/2.) for low, high in zip(bins[:-1], bins[1:])], 2)
    bins = pd.cut(x.mc, bins, labels=labels)
    x[name] = bins
    return x


def get_data(dz, edges=np.arange(-2.25,2.5, .5)):
    if 'conf_0index' not in dz.columns:
        dz.loc[:, 'conf_0index'] = dz.confidence-1

    dlin = dlin = (dz.groupby(['snum', 'session_num', 'block_num'])
         .apply(lambda x: qcut(x, edges, 'lincon'))
         .groupby(['lincon', 'snum', 'noise_sigma', 'session_num', 'block_num'])
         .mean()
         .reset_index())
    dz = dz.copy()
    d2 = qcut(dz, edges, 'lincon').set_index(['lincon', 'snum', 'session_num', 'noise_sigma'])
    d2 = d2.drop(set(d2.columns) - set(['correct', 'conf_0index', 'session_num', 'trial', 'mc']), axis=1)
    dlin = dlin.groupby(['lincon', 'snum', 'session_num', 'noise_sigma']).mean()
    d2 = d2.join(dlin, rsuffix='agg')
    d2 =  d2.reset_index()
    idnan = np.isnan(d2.conf_0index.values) | np.isnan(d2.lincon.values)
    #d2.loc[:, 'rside'] = d2.lincon>0
    #d2 = d2.loc[~idnan, :]
    return d2


def fit_model(formula, data=None, edges=np.arange(-2.25,2.5, .5), iter=2000):
    fit, y, X, subject  = hierarchical_logistic(data, formula, data.snum.values, iter=iter)
    posterior_predictive_check(fit['beta'], X, subject, data, colors=sns.color_palette(n_colors=3))
    return fit, y, X, subject


def hierarchical_logistic(data, formula, iter=2000):

    try:
        sm = pickle.load(open('hmm.stan.pickle'))
    except IOError:
        sm = pystan.StanModel(file='hmm.stan')
        pickle.dump(sm, open('hmm.stan.pickle', 'w'))
    y, X = patsy.dmatrices(formula, data=data)
    subject = np.asarray(patsy.dmatrix('snum -1', data=data)).astype(int).ravel()
    assert(len(y) == len(X) == len(subject))
    datadict = {'x':np.asarray(X), 'y':np.asarray(y).astype(int).ravel(),
        'll':subject.astype(int), 'D':X.shape[1],
        'L':len(np.unique(subject)), 'N':len(y)}

    fit = sm.sampling(data=datadict, iter=iter, chains=2)
    return fit, y, X, subject


def plot(fit, X):
    for i, n in enumerate(X.design_info.column_names):
        hist(fit['mu'][:,i], bins=25, label=n)
        title(n)
    legend()


def pred(betas, X, subject):
    p = expit((betas[subject[:]-1, :]*X).sum(1))
    return binom.rvs(1, p)


def predictions(betas, X, subject):
    return np.vstack([pred(b, X, subject) for b in betas])


def conf_vs_contrast(p, data):
    data.loc[:, 'predicted'] = p
    davg = data.groupby(['lincon', 'noise_sigma']).mean().reset_index()
    t = pd.pivot_table(index='lincon', values='predicted', columns='noise_sigma', data=davg)
    return t.index.values, t.values


def posterior_predictive_check(betas, X, subject, data, colors=sns.color_palette(n_colors=3)):
    prediction = predictions(betas, X, subject)
    y = np.dstack([conf_vs_contrast(p.ravel(), data)[1] for p in prediction])
    x, true = conf_vs_contrast(data.conf_0index.values, data)

    for ns, color in zip(list(range(3)), colors):
        yns = y[:, ns, :]
        ys = np.array([np.percentile(yns[i,:], [2.5, 97.5]) for i in range(y.shape[0])])
        plt.fill_between(x, ys[:, 0], ys[:, 1], facecolor=colors[ns], alpha=0.5)

    for ns, color in zip(list(range(3)), colors):
        plt.plot(x, true[:, ns], color='k', lw=2.5, alpha=0.75)
        print((colors[ns]))
        plt.plot(x, true[:, ns], color=colors[ns], lw=2)

    print('Howdy')
