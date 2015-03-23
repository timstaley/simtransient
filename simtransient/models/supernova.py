from __future__ import absolute_import
import numpy as np
import pandas as pd
from simlightcurve.curves import ModSigmoidExp, Minishell
from simtransient.models.multivariate import MultivarGaussHypers


class Sn1aOpticalEnsemble(MultivarGaussHypers):
    # curve_class = ModSigmoidExp


    # Multivariate gaussian hyperparams:
    default_gauss_pars = pd.DataFrame(index=('mu', 'sigma'))
    default_gauss_pars['a'] = 1.15, 0.15
    default_gauss_pars['rise_tau'] = 3, 0.5
    default_gauss_pars['decay_tau'] = 15, 3

    default_gauss_correlations = {('a', 'rise_tau'): 0.9,
                                  ('a', 'decay_tau'): 0.5,
                                  ('rise_tau', 'decay_tau'): 0.7,
                                  }


    def __init__(self,
                 gauss_pars=default_gauss_pars,
                 gauss_correlations=default_gauss_correlations,
                 ):
        super(Sn1aOpticalEnsemble, self).__init__(
            ModSigmoidExp,
            gauss_pars,
            gauss_correlations,
            fixed_pars={'b': 0.0, 't1_minus_t0': 0.0}
        )

    def evaluate(self, t, a, rise_tau, decay_tau, t0):
        return self.curve_class.evaluate(t,
                                         a,
                                         self.fixed_pars['b'],
                                         self.fixed_pars['t1_minus_t0'],
                                         rise_tau,
                                         decay_tau,
                                         t0
                                         )


class Sn2OpticalEnsemble(MultivarGaussHypers):
    # curve_class = ModSigmoidExp


    # Multivariate gaussian hyperparams:
    default_gauss_pars = pd.DataFrame(index=('mu', 'sigma'))
    default_gauss_pars['a'] = 1.15, 0.15
    default_gauss_pars['b'] = 4e-3, 1.5e-3
    default_gauss_pars['t1_minus_t0'] = 16, 1.5
    default_gauss_pars['rise_tau'] = 3, 0.5
    default_gauss_pars['decay_tau'] = 15, 3

    default_gauss_correlations = {('a', 'rise_tau'): 0.7,
                                  ('a', 'decay_tau'): 0.3,
                                  ('rise_tau', 'decay_tau'): 0.5,
                                  ('b', 't1_minus_t0'): .5,
                                  ('decay_tau', 't1_minus_t0'): 0.7,
                                  }


    def __init__(self,
                 gauss_pars=default_gauss_pars,
                 gauss_correlations=default_gauss_correlations,
                 ):
        super(Sn2OpticalEnsemble, self).__init__(
            ModSigmoidExp,
            gauss_pars,
            gauss_correlations,
            fixed_pars=None
        )

    def gauss_lnprior(self, gauss_par_sample):
        # Block any samples with b<0:
        if gauss_par_sample[1] < 0:
            return -np.inf
        return super(Sn2OpticalEnsemble, self).gauss_lnprior(gauss_par_sample)


    def evaluate(self, t,
                 a,
                 b,
                 t1_minus_t0,
                 rise_tau,
                 decay_tau,
                 t0):
        return self.curve_class.evaluate(t, a, b, t1_minus_t0, rise_tau,
                                         decay_tau, t0)
