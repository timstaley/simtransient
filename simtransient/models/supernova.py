from __future__ import absolute_import
from simlightcurve.curves import ModSigmoidExp, Minishell
from simtransient.utils import build_covariance_matrix, mahalanobis_sq
from collections import OrderedDict
import pandas as pd

class Sn1aOpticalEnsemble(object):
    curve = ModSigmoidExp
    fixed_params = {'b': 0.0, 't1_minus_t0':0.0}

    #Multivariate gaussian params:
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

    # @staticmethod
    # def logprior(a,
    #              b,
    #              t1_minus_t0,
    #              rise_tau,
    #              decay_tau, ):




# class SnType1a(TransientBase):
#     def __init__(self, epoch0):
#         super(SnType1a, self).__init__(epoch0)
#         day = 24*60*60#units of seconds
#         rise_tau= 9*day
#         decay_tau= 25*day
#
#         self._add_lightcurve('optical', 0,
#                              ModSigmoidExp(peak_flux=1, b=0,
#                                             t1_minus_t0=0,
#                                             rise_tau=rise_tau,
#                                             decay_tau=decay_tau,
#                                             )
#         )
#         self._add_lightcurve('radio',0,Null())


# class SnType2(TransientBase):
#     def __init__(self, epoch0):
#         super(SnType2, self).__init__(epoch0)
#         day = 24*60*60#units of seconds
#         rise_tau= 9*day
#         decay_tau= 25*day
#         t1_offset = decay_tau*0.8
#         b=1.5e-13
#         self._add_lightcurve('optical', 0,
#                              ModSigmoidExp(peak_flux=1, b=b,
#                                             t1_minus_t0=t1_offset,
#                                             rise_tau=rise_tau,
#                                             decay_tau=decay_tau,
#                                             )
#         )
#         self._add_lightcurve('radio', timedelta(days=45),
#                              Minishell(k1=2.5e2, k2=1.38e2, k3=1.47e5, beta=-1.5,
#                                            delta1=-2.56, delta2=-2.69)
#         )