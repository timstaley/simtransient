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
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric.bandwidths import bw_silverman


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
        elif obs_sigma is not None:
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
        self.ml_pars[:] = ml_results.x

        neg_post = lambda *args: -self.lnprob(*args)
        self.map_pars = pd.Series(self.init_pars, name='MaxPosterior',
                                  copy=True)
        map_results = minimize(neg_post, self.map_pars)
        self.map_pars[:] = map_results.x
        self.init_pars[:] = self.map_pars


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


    def run(self, sampler, nsteps):
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
                      forecast_marker=True,
                      use_kde=True,
                      kde_noise_sigma=None,
                      plot_data=True,
                      true_curve=None,
                      subsample_size=75,
                      axes=None,
                      palette=None,
                      ):
        """
        Plots the lightcurve ensemble for the given tsteps range.

        If t_forecast is provided, a side-plot is created showing a KDE
        cross-section of the lightcurve densities at that moment.
        """

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
            c_forecast_overplot = palette[4]
            c_true = palette[5]
        else:
            c_trace = palette['trace']
            if plot_data and self.obs_data is not None:
                c_data = palette['data']
            if t_forecast is not None:
                c_forecast_overplot = palette['forecast']
            if true_curve is not None:
                c_true = palette['true']

        alpha_forecast = 0.5
        alpha_data = 0.9
        ls_true = '--'
        ls_xsection = ':'

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

        # if pad_low_margin:
        # # Add a little breathing space at the bottom of the plot:
        #     ylim = ts_ax.get_ylim()
        #     ts_ax.set_ylim(ylim[0] - 0.05*(ylim[1]-ylim[0]), ylim[1])

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
            if hist_ax:
                hist_ax.axhline(true_curve(t_forecast),
                                ls=ls_true,
                                c=c_true)

        if t_forecast is not None:
            forecast_data = self.compute_forecast_data(t_forecast)
            if use_kde:
                kde_ylims = self._plot_t_forecast_kde(
                                        forecast_data,
                                        noise_sigma=kde_noise_sigma,
                                        kde_ax=hist_ax,
                                        c_kde=c_trace)
                if kde_ylims[0] < 0:
                    #Extend the plots downwards to display KDE low tail:
                    ts_ax.set_ylim(kde_ylims[0], ts_ax.get_ylim()[1])
            else:
                self._plot_t_forecast_hist(forecast_data,
                                           hist_ax=hist_ax,
                                           c_hist=c_trace)
            hist_ax.set_ylim(ts_ax.get_ylim())
            if forecast_marker:
                ts_ax.axvline(t_forecast,
                              ls=ls_xsection,
                              lw=lw_overplot,
                              color=c_forecast_overplot,
                              alpha=alpha_forecast,
                              label='Forecast epoch'
                              )


            # ts_ax.axhline(np.mean(forecast_data),
            # label='Forecast',
            # ls=ls_overplot,
            #               lw=lw_overplot,
            #               color=c_forecast_overplot,
            #               alpha=alpha_forecast,
            #               )

        ts_ax.legend(loc='best')
        return ts_ax, hist_ax


    def compute_forecast_data(self, t_forecast):
        forecast_data = np.fromiter(
            (self.ensemble.evaluate(t_forecast, *theta, **self.fixed_pars)
             for theta in self.trimmed),
            dtype=np.float)
        return forecast_data

    def get_kde(self, forecast_data, bandwidth=None):
        kde = KDEUnivariate(forecast_data)
        silverman_bw = bw_silverman(forecast_data)
        if bandwidth is None or bandwidth < silverman_bw:
            kde.fit(bw=silverman_bw)
        else:
            kde.fit(bw=bandwidth)
        return kde


        if noise_sigma is not None and noise_sigma>silverman_bw:
            kde_obs=KDEUnivariate(forecast_data)
            kde_obs.fit(bw=noise_sigma)
            kde_obs = kde_obs.evaluate(y_steps)
            kde_ax.plot(kde_obs, y_steps,
                        c=c_kde, ls='-')


    def _plot_t_forecast_hist(self,
                              forecast_data,
                              c_hist,
                              hist_ax=None,
                              alpha_hist=0.6):

        if hist_ax is None:
            hist_ax = plt.gca()

        hist_ax.hist(forecast_data,
                     orientation='horizontal',
                     normed=True,
                     color=c_hist,
                     alpha=alpha_hist)

        # Prevent ticks overlapping too much:
        max_hist_xticks = 4
        hist_xloc = plt.MaxNLocator(max_hist_xticks)
        hist_ax.xaxis.set_major_locator(hist_xloc)

    def _plot_t_forecast_kde(self,
                             forecast_data,
                             c_kde,
                             noise_sigma=None,
                             n_ysteps=200,
                             kde_ax=None,
                             alpha_kde_fill=0.2):
        if kde_ax is None:
            kde_ax = plt.gca()

        data_minmax = forecast_data.min(), forecast_data.max()
        data_range = (data_minmax[1] - data_minmax[0])
        if noise_sigma is None:
            y_limits = (data_minmax[0] - 0.1 * data_range,
                        data_minmax[1] + 0.1 * data_range)
        else:
            y_limits = (data_minmax[0] - 2.5 * noise_sigma,
                        data_minmax[1] + 2.5 * noise_sigma)
        y_steps = np.linspace(y_limits[0], y_limits[1], n_ysteps)

        kde_intrinsic = self.get_kde(forecast_data)
        kde_intrinsic_xpts = kde_intrinsic.evaluate(y_steps)
        kde_ax.plot(kde_intrinsic_xpts, y_steps,
                    c=c_kde, ls='--')

        if noise_sigma is not None:
            kde_obs=self.get_kde(forecast_data, bandwidth=noise_sigma)
            kde_obs_xpts = kde_obs.evaluate(y_steps)
            kde_ax.plot(kde_obs_xpts, y_steps,
                        c=c_kde, ls='-')
            kde_ax.fill_betweenx(y_steps,
                             0, kde_obs_xpts,
                             alpha=alpha_kde_fill,
                             color=c_kde)
        else:
            kde_ax.fill_betweenx(y_steps,
                             0, kde_intrinsic_xpts,
                             alpha=alpha_kde_fill,
                             color=c_kde)




        # Prevent ticks overlapping too much:
        max_hist_xticks = 4
        kde_xloc = plt.MaxNLocator(max_hist_xticks)
        kde_ax.xaxis.set_major_locator(kde_xloc)
        return y_limits


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
            return -np.inf
        else:
            prob = lp + self.gaussian_lnlikelihood(theta)
        if np.isfinite(prob):
            return prob
        else:
            return -np.inf


