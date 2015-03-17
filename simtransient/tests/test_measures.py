from __future__ import absolute_import
from unittest import TestCase
import numpy as np
from simtransient import measures

class TestUniformLnPrior(TestCase):
    def test_01(self):
        prior1 = measures.get_uniform_lnprior(0,1)
        prior2 = measures.get_uniform_lnprior(1,0)

        for prior in prior1,prior2:
            self.assertTrue(prior(0.5)==0)
            self.assertTrue(prior(0.)==-np.inf)
            self.assertTrue(prior(1.)==-np.inf)

    def test_arbitrary_range(self):
        prior = measures.get_uniform_lnprior(-5, 5)
        self.assertAlmostEqual(np.exp(prior(-4)),1/10.)
        self.assertAlmostEqual(np.exp(prior(4)),1/10.)
        self.assertTrue(prior(5.)==-np.inf)