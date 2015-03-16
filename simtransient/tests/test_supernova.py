from __future__ import absolute_import
from unittest import TestCase
import numpy as np
from scipy.stats import multivariate_normal
from simtransient.models.supernova import Sn1aOpticalEnsemble


class TestSN1a(TestCase):
    def setUp(self):
        ens = self.ens = Sn1aOpticalEnsemble()
        self.scipy_rv = multivariate_normal(mean=ens.gpars.loc['mu'],
                                            cov=ens.gpar_cov)

    def test_gpar_prior(self):
        test_samples = self.scipy_rv.rvs(int(1e5))
        scipy_logpdfs = self.scipy_rv.logpdf(test_samples)
        lnpriors = self.ens.gpar_lnprior(test_samples)
        self.assertTrue(np.allclose(scipy_logpdfs,
                                    lnpriors))