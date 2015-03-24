import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
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

    def plot_hists(self, axes=None):
        stplot.chain.all_hists(self.sampler.chain, self.chainstats,
                               self.free_par_names, axes=axes)


    def plot_triangle(self, axes=None,
                      truths=None):
        _ = triangle.corner(self.trimmed,
                            labels=self.free_par_names,
                            quantiles=[0.05, 0.5, 0.95],
                            truths=truths)


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
                hist_ax=None
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
            c_trace=palette['trace']
            if plot_data and self.obs_data is not None:
                c_data=palette['data']
            if t_forecast is not None:
                c_forecast = palette['forecast']
            if true_curve is not None:
                c_true = palette['true']


        alpha_forecast=0.5
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
                            zorder=5, alpha=0.8)

        if true_curve is not None:
            ts_ax.plot(tsteps, true_curve(tsteps), ls='--', c=c_true,
                       label='True',
                       lw=lw_overplot)


        if t_forecast is not None:
            forecast_data =  np.fromiter(
                ( self.ensemble.evaluate(t_forecast, *theta, **self.fixed_pars)
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
            if true_curve:
                hist_ax.axhline(true_curve(t_forecast),
                                ls=ls_overplot,
                                c=c_true)

            ts_ax.legend(loc='best')
        return ts_ax,hist_ax
