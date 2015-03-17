"""
Generic priors and likelihood functions.
"""
from __future__ import absolute_import
import numpy as np

def gauss_lnlikelihood(model_pars, model_ensemble, obs_data, obs_sigma):
    """
    Model likelihood assuming unbiased Gaussian noise of width ``obs_sigma``.

    obs_sigma can be scalar or an array matching obs_data
    (latter needs testing but I think it's correct)

    ..math:

        -0.5 \sum_{i=1}^N \left[ ln(2\pi\sigma_i^2) +
                                ((x_i - \alpha_i)/\sigma_i)^2  \right]


    """
    intrinsic_fluxes = model_ensemble.evaluate(obs_data.index, *model_pars)
    return -0.5 * np.sum(np.log(2 * np.pi * obs_sigma ** 2) +
                         ((obs_data-intrinsic_fluxes) /obs_sigma) ** 2)


def get_uniform_lnprior(x1, x2):
    xmin = min(x1, x2)
    xmax = max(x1, x2)
    uniform_value = -np.log(xmax - xmin)

    def uniform_prior(x):
        if xmin < x < xmax:
            return uniform_value
        else:
            return -np.inf

    return uniform_prior