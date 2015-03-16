from __future__ import absolute_import
from simlightcurve.curves import ModSigmoidExp, Minishell
from simtransient.utils import build_covariance_matrix, mahalanobis_sq
from collections import OrderedDict
import numpy as np
import pandas as pd

class Sn1aOpticalEnsemble(object):
    curve_class = ModSigmoidExp
    fixed_params = {'b': 0.0, 't1_minus_t0':0.0}

    #Multivariate gaussian hyperparams:
    gpars = pd.DataFrame(index=('mu','sigma'))
    gpars['a'] = 1.15, 0.15
    gpars['rise_tau']= 3, 0.5
    gpars['decay_tau'] = 15, 3

    gpar_corrs = {('a','rise_tau'):0.9,
                  ('a','decay_tau'):0.5,
                  ('rise_tau','decay_tau'):0.7,
                  }


    def __init__(self):
        self.gpar_cov = build_covariance_matrix(self.gpars.loc['sigma'],
                                                self.gpar_corrs)
        self.gpar_icov = np.linalg.inv(self.gpar_cov)
        ndim=len(self.gpars.T)
        icov_det = np.linalg.det(self.gpar_icov)
        c = np.sqrt(((2 * np.pi) ** ndim) * np.linalg.det(self.gpar_cov))
        self._lnprior_offset= -np.log(c)


    def lnprior(self, params):
        """
        Since this is a prior, in the case that params is not a vector
        but a 2-d array (i.e. a stack of vectors), then we use broadcasting
        to calculate the lnprior for each vector of hyperparameters in turn.

        (Contrast with the log-likelihood for multiple measurements of a
         multivariate Gaussian, where we should sum the results)
        """
        # def lnprob(mu,x,icov):
        mu = self.gpars.loc['mu']
        diff = params-mu.values
        raw_value = -0.5*(mahalanobis_sq(self.gpar_icov,diff))
        return self._lnprior_offset + raw_value

    def _get_eval_pars(self, gpars, t0):
        epars = dict(a=gpars[0],rise_tau=gpars[1],decay_tau=gpars[2],
                      t0=t0)
        epars.update(self.fixed_params)
        return epars


    def get_curve(self, a, rise_tau, decay_tau, t0=0):
        kwargs = dict(a=a,rise_tau=rise_tau,decay_tau=decay_tau,
                      t0=t0)
        curve_params = self.fixed_params.copy()
        curve_params.update(kwargs)
        # print curve_params
        return self.curve_class(**curve_params)

    def evaluate(self, t, a,rise_tau,decay_tau, t0):
        epars = self._get_eval_pars((a,rise_tau,decay_tau),t0)
        return self.curve_class.evaluate(t,**epars)
