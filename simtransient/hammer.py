"""
Convenience functions for wrapping emcee (an emcee rapper, geddit)
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import emcee
import logging

logger = logging.getLogger(__name__)

def prep_ensemble_sampler(init_params,
                          lnprob,
                          args,
                          nwalkers,
                          nthreads,
                          a=2,
                          ballsize=1e-4):
    ndim = len(init_params)  # number of parameters in the model

    sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                    lnprob,
                                    a=a,
                                    args=args,
                                    threads=nthreads)

    initpar_ball = ( init_params +
                     ballsize * np.random.randn(nwalkers * ndim).reshape(
                         nwalkers, ndim))

    return sampler, initpar_ball


def prep_pt_sampler(init_params,
                    lnprior,
                    lnlikelihood,
                    lnlikeargs,
                    nwalkers,
                    nthreads,
                    a=2,
                    ntemps=10,
                    ballsize=1e-4):
    ndim = len(init_params)  # number of parameters in the model

    sampler = emcee.PTSampler(ntemps, nwalkers, ndim,
                              logp=lnprior,
                              logl=lnlikelihood,
                              loglargs=lnlikeargs,
                              a=a,
                              threads=nthreads)

    gaussian_ball = np.random.randn(ndim * nwalkers * ntemps).reshape(
        ntemps, nwalkers, ndim)
    initpar_ball = ( init_params + ballsize * gaussian_ball)

    return sampler, initpar_ball


def trim_chain(sampler, pt=False,
               burn_length=2):
    if not pt:
        acceptance = np.median(sampler.acceptance_fraction)
        acorr = np.ceil(np.max(sampler.get_autocorr_time()))
    else:
        #Parallel tempered, take values from lowest temp
        acceptance = np.median(sampler.acceptance_fraction[0])
        acorr = np.ceil(np.max(sampler.get_autocorr_time()[0]))

    if acceptance < 0.15 or acceptance > 0.8:
        logger.warn('Extreme acceptance rate:{}'.format(acceptance))

    nburn = np.ceil(burn_length * acorr)

    chainstats = pd.Series(name='chainstats')
    chainstats['acorr'] = acorr
    chainstats['acceptance'] = acceptance
    chainstats['nburn'] = nburn

    ndim = sampler.chain.shape[-1]
    # Chain index is (walker, step, param)
    # Grab the burned and thinned steps:
    trimmed_samples = sampler.chain[:, nburn::acorr, :]
    #Now chain them all up into a long list of param samples:
    trimmed_samples = trimmed_samples.ravel().reshape(-1, ndim)

    return chainstats, trimmed_samples

def plot_single_param_chain(rawchain,
                            chainstats,
                            param_idx,
                            param_name=None,
                            axes=None):
    if param_name is None:
        param_name=''
    if axes is None:
        fig, axes = plt.subplots(nrows=2, ncols=1)
    ax0, ax1 = axes
    cs = chainstats

    ax0.set_title('{} (Raw)'.format(param_name))
    ax0.set_ylabel(param_name)
    for walker in rawchain[:,:,param_idx]:
        ax0.plot(walker)
    ax0.axvline(cs.nburn, ls=':', color='k')

    ax1.set_title('{} (Thinned)'.format(param_name))
    ax1.set_ylabel(param_name)
    # plt.xlabel("Sample")
    for walker in rawchain[:,::cs.acorr,param_idx]:
        ax1.plot(walker)
    ax1.axvline(cs.nburn/cs.acorr, ls=':', color='k')
    ax1.get_yaxis().set_major_locator(mpl.ticker.MaxNLocator(integer=True))

def plot_all_param_chain(rawchain,
                         chainstats,
                         param_names,
                         axes=None):
    ndim = len(param_names)
    if axes is None:
        fig, axes = plt.subplots(nrows=ndim, ncols=2)

    for idx, name in enumerate(param_names):
        ax_pair = axes[idx,:]
        plot_single_param_chain(rawchain,
                                      chainstats,
                                      idx,
                                      name,
                                      axes=ax_pair
                                      )

