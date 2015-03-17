from __future__ import absolute_import
from unittest import TestCase
import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.distance import mahalanobis
from simtransient import utils


class TestMahalanobis(TestCase):
    def setUp(self):
        #A quick-and-dirty ndim covariance, courtesy of the emcee docs.
        ndim = 10
        self.means = np.random.rand(ndim)
        cov = 0.5 - np.random.rand(ndim ** 2).reshape((ndim, ndim))
        cov = np.triu(cov)
        cov += cov.T - np.diag(cov.diagonal())
        self.cov = np.dot(cov,cov)
        self.icov = np.linalg.inv(self.cov)

        self.rv = multivariate_normal(mean = self.means,
                                      cov=self.cov)
        self.n_samples = int(1e4)
        self.sample = self.rv.rvs(self.n_samples)

    def test_scipy_equality(self):
        diff = self.sample - self.means
        simtransient_results = utils.mahalanobis_sq(self.icov, diff)
        scipy_results = np.zeros_like(simtransient_results)
        for idx, s in enumerate(self.sample):
            scipy_results[idx] = mahalanobis(self.means, s, self.icov)**2

        self.assertTrue(np.allclose(scipy_results, simtransient_results))


class TestUniformLnPrior(TestCase):
    def test_01(self):
        prior1 = utils.get_uniform_lnprior(0,1)
        prior2 = utils.get_uniform_lnprior(1,0)

        for prior in prior1,prior2:
            self.assertTrue(prior(0.5)==0)
            self.assertTrue(prior(0.)==-np.inf)
            self.assertTrue(prior(1.)==-np.inf)

    def test_arbitrary_range(self):
        prior = utils.get_uniform_lnprior(-5, 5)
        self.assertAlmostEqual(np.exp(prior(-4)),1/10.)
        self.assertAlmostEqual(np.exp(prior(4)),1/10.)
        self.assertTrue(prior(5.)==-np.inf)