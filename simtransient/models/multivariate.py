from __future__ import absolute_import
from abc import ABCMeta, abstractmethod
import numpy as np
from simtransient.utils import build_covariance_matrix, mahalanobis_sq
import logging

logger = logging.getLogger(__name__)



class MultivarGaussHypers(object):
    """
    Lightcurve ensemble with mostly multivariate Gaussian hyperparams.

    "Mostly" because an exception is made for the (usually flat-prior) t0
    hyperparam, and fixed hyperparams are treated separately.

    Note the abstract ``evaluate`` method that should be implemented downstream.
    """
    __metaclass__ = ABCMeta

    def __init__(self,
                 curve_class,
                 gauss_pars,
                 gauss_correlations,
                 fixed_pars=None
                 ):
        """
        Initialization.

        Also builds the covariance matrix for the multivariate Gaussian (MVG)
        hyperparams, given their variances and correlations.

        Args:
            curve_class : simlightcurve curve class.
            gauss_pars (pandas.DataFrame): Dataframe composed of a Series
                for each MVG hyperparam, entries 'mu' and 'sigma' for mean
                and standard deviation in each.
            gauss_correlations (dict): Maps param-name pair-tuples to
                correlation coefficient values.
                Any missing pairs are equivalent to a zero-value correlation.
            fixed_pars (dict): Maps fixed param names to their fixed values.
        """

        if fixed_pars is None:
            fixed_pars={}
        # Sanity check:
        # We expect to supply the curve_class with
        # generic Gaussian + fixed params, plus t0
        assert (len(gauss_pars.keys()) + len(fixed_pars) + 1 ==
                len(curve_class.param_names))

        self.curve_class = curve_class
        self.gauss_pars=gauss_pars
        self.gauss_correlations=gauss_correlations
        self.fixed_pars=fixed_pars

        #Calculate various properties of the Gaussian hyperparams:
        self.gauss_cov = build_covariance_matrix(self.gauss_pars.T.sigma,
                                                 self.gauss_correlations)
        try:
            np.linalg.cholesky(self.gauss_cov)
        except np.linalg.LinAlgError:
            logger.error("Provided correlation values result in a covariance"
                         " which is not positive semidefinite.")
            raise

        self.gauss_icov = np.linalg.inv(self.gauss_cov)
        ndim_gauss = len(self.gauss_pars.T)
        c = np.sqrt(((2 * np.pi) ** ndim_gauss) * np.linalg.det(self.gauss_cov))
        self._gauss_lnprior_offset = -np.log(c)


    def get_curve(self, **kwargs):
        curve_params = self.fixed_pars.copy()
        curve_params.update(**kwargs)
        # print curve_params
        return self.curve_class(**curve_params)

    @abstractmethod
    def evaluate(self, tsteps, gauss_pars, t0):
        """
        Defines a (preferably efficient) wrapper about curve_class.evaluate.

        Should merge the free parameters with any fixed params that the
        underlying evaluate call requires.

        NB This is functionally equivalent to get_curve(free_params,t0)(tsteps),
        but that requires instantiating the curve_class for each call so it's
        much slower.
        """

    def t0_lnprior(self, t0_sample):
        """
        Default flat prior on t0, can be dynamically replaced.

        So e.g. you can over-ride this with a bounded uniform prior
        once you know approximately where your lightcurve should be.

        Args:
            t0_sample (scalar or array): Values for calculating the lnprior.
        """
        return np.zeros_like(t0_sample)

    def gauss_lnprior(self, gauss_par_sample):
        """
        Calculate the multivariate Gauss prior.

        Since this is a prior, in the case that gauss_par_sample is not a vector
        but a 2-d array (i.e. a stack of vectors), then we use broadcasting
        to calculate the lnprior for each vector of hyperparameters in turn.

        (Contrast with the log-likelihood for multiple measurements of a
         multivariate Gaussian, where we should sum the results)
        """
        # def lnprob(mu,x,icov):
        mu = self.gauss_pars.T.mu
        diff = gauss_par_sample - mu.values
        raw_value = -0.5 * (mahalanobis_sq(self.gauss_icov, diff))
        return self._gauss_lnprior_offset + raw_value

    def lnprior(self, model_pars):
        return ( self.gauss_lnprior(model_pars[:-1]) +
                 self.t0_lnprior(model_pars[-1])
                 )


