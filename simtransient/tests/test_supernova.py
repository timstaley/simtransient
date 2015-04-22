from __future__ import absolute_import
from unittest import TestCase
import numpy as np
from scipy.stats import multivariate_normal
from simtransient.models.supernova import Sn1aOpticalEnsemble


class TestSN1a(TestCase):
    def setUp(self):
        ens = self.ens = Sn1aOpticalEnsemble()
        self.scipy_rv = multivariate_normal(mean=ens.gauss_pars.loc['mu'],
                                            cov=ens.gauss_cov)

    def test_gpar_prior(self):
        """
        Test the prior function with a single sample.
        """
        random_param_vec = self.scipy_rv.rvs()
        # Must force the first parameter to be positive, or result may
        # not match that returned by scipy multivariate_normal.
        good_param_vec = random_param_vec.copy()
        good_param_vec[0] = self.ens.gauss_pars.a.mu
        self.assertAlmostEqual(self.scipy_rv.logpdf(good_param_vec),
                               self.ens.gauss_lnprior(good_param_vec))

        # We don't allow negative amplitude for this model:
        invalid_param_vec = random_param_vec.copy()
        invalid_param_vec[0] = -0.1
        self.assertEqual(-np.inf,
                         self.ens.gauss_lnprior(invalid_param_vec))


    def test_gpar_prior_vectorized(self):
        """
        Test the prior function works for stack of samples.
        """
        test_samples = self.scipy_rv.rvs(int(1e5))
        scipy_logpdfs = self.scipy_rv.logpdf(test_samples)
        lnpriors = self.ens.gauss_lnprior(test_samples)
        for idx in xrange(len(test_samples)):
            if test_samples[idx][0]>0:
                self.assertAlmostEqual(scipy_logpdfs[idx],lnpriors[idx])
            else:
                self.assertEqual(lnpriors[idx], -np.inf)