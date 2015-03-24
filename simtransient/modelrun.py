import numpy as np
import emcee
from simtransient.models.multivariate import MultivarGaussHypers
import simtransient.hammer as hammer
import simtransient.plot as stplot
import triangle


class ModelRun(object):
    def __init__(self,
                 ensemble,
                 obs_data=None,
                 obs_sigma=None,
                 nwalkers=100,
                 init_pars=None,
                 init_par_ball=5e-4,
                 use_pt=False,
                 ntemps=10,
                 emcee_kwargs=None,
                 ):

        assert isinstance(ensemble, MultivarGaussHypers)
        self.ensemble = ensemble
        self.obs_data = obs_data
        self.obs_sigma = obs_sigma
        self.pt = use_pt

        self.chainstats = None
        self.trimmed = None  # Burned and thinned samples

        if emcee_kwargs is None:
            emcee_kwargs = {}

        if obs_data is None:
            self.free_par_names = ensemble.gauss_pars.keys()
            self.fixed_pars = {'t0': 0}
            if init_pars is None:
                self.init_pars = ensemble.gauss_pars.T.mu.values
            self.lnprior = ensemble.gauss_lnprior
            self.lnprob = ensemble.gauss_lnprior
            self.lnlike_args = []
            self.ndim = len(self.init_pars)

        if not self.pt:
            self.sampler = emcee.EnsembleSampler(nwalkers,
                                                 self.ndim,
                                                 self.lnprob,
                                                 args=self.lnlike_args,
                                                 **emcee_kwargs
                                                 )
            if np.isscalar(init_par_ball):
                self.init_par_ball = ( self.init_pars +
                                       init_par_ball * np.random.randn(
                                           nwalkers * self.ndim).reshape(
                                           nwalkers, self.ndim))

    def sample(self, nsteps):
        _ = self.sampler.run_mcmc(self.init_par_ball, N=nsteps)
        self.chainstats, self.trimmed = hammer.trim_chain(self.sampler, self.pt)

    def plot_walkers(self, axes=None):
        stplot.chain.all_walkers(self.sampler.chain, self.chainstats,
                                 self.free_par_names, axes)


    def triangle_plot(self, axes=None,
                      truths=None):
        _=triangle.corner(self.trimmed,
                 labels=self.free_par_names,
                 quantiles=[0.05, 0.5, 0.95],
                 truths=truths)



