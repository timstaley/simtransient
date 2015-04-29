import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.optimize import minimize
import emcee
import triangle
import seaborn
from simtransient.models.multivariate import MultivarGaussHypers
import simtransient.hammer as hammer
import simtransient.plot as stplot
import simtransient.utils as stutils

class ModelRun(object):
    def __init__(self,
                 ensemble,
                 obs_data=None,
                 obs_sigma=None,
                 nwalkers=100,
                 init_pars=None,
                 init_pars_ball=None,
                 init_pars_ballsize=5e-4,
                 use_pt=False,
                 ntemps=10,
                 emcee_kwargs=None,
                 ):

        assert isinstance(ensemble, MultivarGaussHypers)
        self.ensemble = ensemble
        self.obs_data = obs_data

        if np.isscalar(obs_sigma):
            self.obs_sigma = obs_sigma * np.ones_like(obs_data)
            self.obs_sigma_sq = obs_sigma * np.ones_like(obs_data)
        else:
            self.obs_sigma = np.array(obs_sigma)
            self.obs_sigma_sq = np.power(self.obs_sigma, 2)

        self.nwalkers = nwalkers
        self.ntemps = ntemps

        self.pt = use_pt
        self.chainstats = None
        self.trimmed = None  # Burned and thinned samples

        if emcee_kwargs is None:
            emcee_kwargs = {}

        self.init_pars = init_pars
        self.init_pars_ball = init_pars_ball
        self.init_pars_ballsize = init_pars_ballsize
        self.ml_pars = None
        self.map_pars = None

        if init_pars is None:
            self.set_init_pars_based_on_priors()
        self.ndim = len(self.init_pars)

        if obs_data is None:
            self.lnprior = ensemble.gauss_lnprior
            self.lnprob = ensemble.gauss_lnprior
            self.lnlike_args = []
        else:
            self.lnprior = ensemble.lnprior
            self.lnlike = self.gaussian_lnlikelihood
            self.lnlike_args = []
            self.lnprob = self.gaussian_lnprob

        self.set_init_par_ball()



    def get_sampler(self, **kwargs):
        if not self.pt:
            return emcee.EnsembleSampler(self.nwalkers,
                                         self.ndim,
                                         self.lnprob,
                                         args=self.lnlike_args,
                                         **kwargs
                                         )
        else:
            return emcee.PTSampler(self.ntemps,
                                   self.nwalkers,
                                   self.ndim,
                                   logl=self.lnlike,
                                   logp=self.lnprior,
                                   loglargs=self.lnlike_args,
                                   **kwargs
                                   )


    @property
    def postchain(self):
        """
        Get the the posterior sample chain.
        """
        if not self.pt:
            return self._chain
        else:
            return self._chain[0]

    @property
    def init_curve(self):
        return self.ensemble.get_curve(**self.init_pars)

    @property
    def ml_curve(self):
        return self.ensemble.get_curve(**self.ml_pars)

    @property
    def map_curve(self):
        return self.ensemble.get_curve(**self.map_pars)


    def set_init_pars_based_on_priors(self, t0=None):
        self.free_par_names = list(self.ensemble.gauss_pars)
        self.init_pars = pd.Series(data=self.ensemble.gauss_pars.T.mu,
                                   name="InitParams")
        if self.obs_data is None:
            if t0 is None:
                t0 = 0
            self.fixed_pars = {'t0': t0}
        else:
            self.free_par_names.append('t0')
            if t0 is None:
                t0 = self.obs_data.index[0]
            self.init_pars['t0'] = t0
            self.fixed_pars = {}


    def fit_data(self):
        if self.obs_data is None:
            raise RuntimeError("No data to fit!")

        neg_likelihood = lambda *args: -self.lnlike(*args)
        self.ml_pars = pd.Series(self.init_pars, name='MaxLikelihood',
                                 copy=True)
        ml_results = minimize(neg_likelihood, self.ml_pars)
        self.ml_pars[:]=ml_results.x

        neg_post = lambda *args: -self.lnprob(*args)
        self.map_pars = pd.Series(self.init_pars, name='MaxPosterior',
                                  copy=True)
        map_results = minimize(neg_post, self.map_pars)
        self.map_pars[:]=map_results.x
        self.init_pars[:]=self.map_pars


    def set_init_par_ball(self):
        if not self.pt:
            self.init_par_ball = ( self.init_pars.values +
                                   self.init_pars_ballsize * np.random.randn(
                                       self.nwalkers * self.ndim).reshape(
                                       self.nwalkers, self.ndim))
        else:
            self.init_par_ball = ( self.init_pars.values +
                                   self.init_pars_ballsize * np.random.randn(
                                       self.ntemps * self.nwalkers * self.ndim).reshape(
                                       self.ntemps, self.nwalkers,
                                       self.ndim))


    def run(self, sampler,nsteps):
        pos, lnprob, rstate = sampler.run_mcmc(self.init_par_ball,
                                                    N=nsteps)
        self.init_par_ball = pos
        self.chainstats, self.trimmed = hammer.trim_chain(sampler, self.pt)
        self._chain = sampler.chain
        return pos, lnprob, rstate



    def plot_walkers(self, axes=None):
        stplot.chain.all_walkers(self.postchain, self.chainstats,
                                 self.free_par_names, axes)


    def plot_hists(self, axes=None):
        stplot.chain.all_hists(self.postchain, self.chainstats,
                               self.free_par_names, axes=axes)


    def plot_triangle(self,
                      **extra_triangle_plot_kwargs
                      ):
        _ = triangle.corner(self.trimmed,
                            labels=self.free_par_names,
                            **extra_triangle_plot_kwargs)


    def plot_forecast(self,
                      tsteps,
                      t_forecast=None,
                      plot_data=True,
                      true_curve=None,
                      subsample_size=75,
                      axes=None,
                      palette=None,
                      alpha_hist=0.6,
                      ):


        if axes is None:
            if t_forecast is None:
                ts_ax = plt.gca()
                hist_ax = None
            else:
                gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
                ts_ax = plt.subplot(gs[0])
                hist_ax = plt.subplot(gs[1])
        else:
            ts_ax, hist_ax = axes

        if palette is None:
            palette = seaborn.color_palette('Set1', 6)
            c_trace = palette[1]
            c_data = palette[2]
            c_forecast = palette[4]
            c_true = palette[5]
        else:
            c_trace = palette['trace']
            if plot_data and self.obs_data is not None:
                c_data = palette['data']
            if t_forecast is not None:
                c_forecast = palette['forecast']
            if true_curve is not None:
                c_true = palette['true']

        alpha_forecast = 0.5
        alpha_data = 0.9
        ls_overplot = '--'
        lw_overplot = 5

        data_ms = 25

        ss_curves = stutils.subsample_curves(self.ensemble,
                                             self.trimmed,
                                             tsteps,
                                             size=subsample_size,
                                             fixed_pars=self.fixed_pars
                                             )

        seaborn.tsplot(ss_curves, tsteps,
                       err_style="unit_traces",
                       ax=ts_ax,
                       ls='',
                       color=c_trace,
                       # condition='Samples'
                       )

        if plot_data and self.obs_data is not None:
            stplot.curve.graded_errorbar(self.obs_data,
                                         self.obs_sigma,
                                         label='Observations',
                                         ms=data_ms,
                                         ax=ts_ax,
                                         color=c_data,
                                         zorder=5,
                                         alpha=alpha_data)

        if true_curve is not None:
            ts_ax.plot(tsteps, true_curve(tsteps), ls='--', c=c_true,
                       label='True',
                       lw=lw_overplot)

        if t_forecast is not None:
            forecast_data = np.fromiter(
                (self.ensemble.evaluate(t_forecast, *theta, **self.fixed_pars)
                 for theta in self.trimmed),
                dtype=np.float)

            ts_ax.axvline(t_forecast,
                          ls=ls_overplot,
                          lw=lw_overplot,
                          color=c_forecast,
                          alpha=alpha_forecast,
                          )
            ts_ax.axhline(np.mean(forecast_data),
                          label='Forecast',
                          ls=ls_overplot,
                          lw=lw_overplot,
                          color=c_forecast,
                          alpha=alpha_forecast,
                          )

            hist_ax.axhline(np.mean(forecast_data),
                            ls=ls_overplot,
                            lw=lw_overplot,
                            color=c_forecast,
                            alpha=alpha_forecast,
                            )
            hist_ax.hist(forecast_data,
                         orientation='horizontal',
                         normed=True,
                         color=c_trace,
                         alpha=alpha_hist)
            _ = hist_ax.set_ylim(ts_ax.get_ylim())
            hist_xlim=hist_ax.get_xlim()

            #Prevent ticks overlapping too much:
            max_hist_xticks = 4
            hist_xloc = plt.MaxNLocator(max_hist_xticks)
            hist_ax.xaxis.set_major_locator(hist_xloc)

        
            if true_curve:
                hist_ax.axhline(true_curve(t_forecast),
                                ls=ls_overplot,
                                c=c_true)

            ts_ax.legend(loc='best')
        return ts_ax, hist_ax


    def gaussian_lnlikelihood(self, theta):
        """
        Model likelihood assuming unbiased Gaussian noise of width ``obs_sigma``.

        obs_sigma can be scalar or an array matching obs_data
        (latter needs testing but I think it's correct)

        ..math:

            -0.5 \sum_{i=1}^N \left[ ln(2\pi\sigma_i^2) +
                                    ((x_i - \alpha_i)/\sigma_i)^2  \right]


        """
        intrinsic_fluxes = self.ensemble.evaluate(self.obs_data.index.values,
                                                  *theta)
        return -0.5 * np.sum(np.log(2 * np.pi * self.obs_sigma_sq) +
                             ((
                                  self.obs_data.values - intrinsic_fluxes) / self.obs_sigma) ** 2)


    def gaussian_lnprob(self, theta):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            prob = -np.inf
        else:
            prob = lp + self.gaussian_lnlikelihood(theta)
        return prob


