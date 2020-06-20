import numpy as np
import os
import pandas as pd


from conf_analysis.behavior import metadata
from joblib import Memory


if "RRZ_LOCAL_TMPDIR" in os.environ.keys():
    memory = Memory(location=os.environ["RRZ_LOCAL_TMPDIR"], verbose=-1)
if "TMPDIR" in os.environ.keys():
    tmpdir = os.environ["TMPDIR"]
    memory = Memory(location=tmpdir, verbose=-1)
else:
    memory = Memory(location=metadata.cachedir, verbose=-1)




@memory.cache
def auc_get_sig_cluster_posterior(data):
    """
    Get uncertainty around average mean by means of bayesian 
    inference. For AUC values, nothing else.

    """
    import pymc3 as pm

    data = data.T
    n_obs, n_clust, n_sig = data.shape
    s = (n_clust, n_sig)
    if np.isnan(data).sum()>0:
        print('Missing values in data. Setting to MA / will be imputed')
        data = np.ma.masked_invalid(data)
    with pm.Model() as model:
        mu_ind = pm.Normal("mu_ind", 0.5, 1, shape=s)
        std_ind = pm.Uniform("std_ind", lower=0, upper=5, shape=s)
        v = pm.Exponential("ν_minus_one", 1 / 29.0, shape=s) + 1
        pm.StudentT(
            "Out_ind", mu=mu_ind, lam=std_ind ** -2, nu=v, shape=s, observed=data
        )

        k = pm.sample(tune=1500, draws=1500)
    return k, model

#sub = 7 cluster=17 sig=1

@memory.cache
def auc_get_sig_cluster_group_diff_posterior(dataA, dataB):
    """
    Get uncertainty around average mean by means of bayesian 
    inference. For AUC values, nothing else.
    """
    import pymc3 as pm

    dataA = dataA.T
    dataB = dataB.T
    n_obs, n_clust, n_sig = dataA.shape
    s = (n_clust, n_sig)        
    if np.isnan(dataA).sum()>0:
        print('Missing values in data. Setting to MA / will be imputed')
        dataA = np.ma.masked_invalid(dataA)
    if np.isnan(dataB).sum()>0:
        print('Missing values in data. Setting to MA / will be imputed')
        dataB = np.ma.masked_invalid(dataB)
    with pm.Model() as model:
        muA = pm.Normal("mu_diff", 0.5, 1, shape=s)
        stdA = pm.Uniform("stdA", lower=0, upper=5, shape=s)
        
        v = pm.Exponential("ν_minus_one", 1 / 29.0, shape=s) + 1
        pm.StudentT(
            "Out_diff", mu=muA, lam=stdA ** -2, nu=v, shape=s, observed=dataB-dataA
        )                
        k = pm.sample(tune=1500, draws=1500)
    return k, model
  