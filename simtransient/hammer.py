"""
Convenience functions for wrapping emcee (an emcee rapper, geddit)
"""
import numpy as np
import pandas as pd
import emcee
import logging

logger = logging.getLogger(__name__)


def prep_ensemble_sampler(init_params,
                          lnprob,
                          args,
                          nwalkers,
                          # nthreads,
                          # a=2,
                          ballsize=1e-4,
                          **kwargs):
    ndim = len(init_params)  # number of parameters in the model

    sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                    lnprob,
                                    args=args,
                                    **kwargs
                                    )

    initpar_ball = ( init_params +
                     ballsize * np.random.randn(nwalkers * ndim).reshape(
                         nwalkers, ndim))

    return sampler, initpar_ball


def prep_pt_sampler(init_params,
                    lnprior,
                    lnlikelihood,
                    lnlikeargs,
                    nwalkers,
                    # nthreads,
                    # a=2,
                    ntemps=10,
                    ballsize=1e-4,
                    **kwargs):
    ndim = len(init_params)  # number of parameters in the model

    sampler = emcee.PTSampler(ntemps, nwalkers, ndim,
                              logp=lnprior,
                              logl=lnlikelihood,
                              loglargs=lnlikeargs,
                              **kwargs)

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
        # Parallel tempered, take values from lowest temp
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
    if not pt:
        trimmed_samples = sampler.chain[:, nburn::acorr, :]
    else:
        trimmed_samples = sampler.chain[0,:, nburn::acorr, :]
    # Now chain them all up into a long list of param samples:
    trimmed_samples = trimmed_samples.ravel().reshape(-1, ndim)

    return chainstats, trimmed_samples

