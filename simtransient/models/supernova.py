from __future__ import absolute_import
from datetime import timedelta

from simtransient.transient import TransientBase
from simlightcurve.curves import ModSigmoidExp, Minishell


class SnType1a(TransientBase):
    def __init__(self, epoch0):
        super(SnType1a, self).__init__(epoch0)
        day = 24*60*60#units of seconds
        rise_tau= 9*day
        decay_tau= 25*day

        self._add_lightcurve('optical', 0,
                             ModSigmoidExp(peak_flux=1, b=0,
                                            t1_minus_t0=0,
                                            rise_tau=rise_tau,
                                            decay_tau=decay_tau,
                                            )
        )
        self._add_lightcurve('radio',0,Null())


class SnType2(TransientBase):
    def __init__(self, epoch0):
        super(SnType2, self).__init__(epoch0)
        day = 24*60*60#units of seconds
        rise_tau= 9*day
        decay_tau= 25*day
        t1_offset = decay_tau*0.8
        b=1.5e-13
        self._add_lightcurve('optical', 0,
                             ModSigmoidExp(peak_flux=1, b=b,
                                            t1_minus_t0=t1_offset,
                                            rise_tau=rise_tau,
                                            decay_tau=decay_tau,
                                            )
        )
        self._add_lightcurve('radio', timedelta(days=45),
                             Minishell(k1=2.5e2, k2=1.38e2, k3=1.47e5, beta=-1.5,
                                           delta1=-2.56, delta2=-2.69)
        )