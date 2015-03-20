"""
Plots for displaying lightcurves (and observation data)
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn
from simtransient.utils import subsample_curves, forecast_ensemble


def graded_errorbar(obs_data,
                    obs_sigma,
                    color='g',
                    base_elw=1.5,
                    ms=16,
                    ax=None,
                    **kwargs):
    if ax is None:
        ax = plt.gca()

    ax.errorbar(obs_data.index, obs_data,
                c=color,
                linewidth=0,
                elinewidth=0,
                ms=ms,
                marker='.',
                **kwargs
                )
    if obs_sigma:
        ax.errorbar(obs_data.index, obs_data, yerr=1 * obs_sigma,
                    c=color,
                    linewidth=0,
                    elinewidth=base_elw * 3,
                    ms=0,
                    marker='.',
                    **kwargs
                    )
        ax.errorbar(obs_data.index, obs_data, yerr=2 * obs_sigma,
                    c=color,
                    linewidth=0,
                    elinewidth=base_elw,
                    mfc=color,
                    ms=0,
                    marker='',
                    **kwargs
                    )
        # ax.scatter(obs_data.index, obs_data, s=50, c=[color])


def forecast_plot(ensemble, samples, tsteps,
                  t_forecast=None,
                  true_curve=None,
                  obs_data=None, obs_sigma=None,
                  subsample_size=100,
                  axes=None,
                  palette=None,
                  alpha_forecast=0.5):
    if axes is None:
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ts_ax = plt.subplot(gs[0])
        hist_ax = plt.subplot(gs[1])
    else:
        ts_ax, hist_ax = axes

    if palette is None:
        palette = seaborn.color_palette('Set1', 6)

    c_trace = palette[1]
    c_true = palette[5]
    c_data = palette[2]
    c_forecast = palette[4]

    ls_overplot = '--'
    lw_overplot = 5

    data_ms = 25

    ss_curves = subsample_curves(ensemble, samples, tsteps)
    seaborn.tsplot(ss_curves, tsteps,
                   err_style="unit_traces",
                   ax=ts_ax, ls='',
                   color=c_trace)

    if obs_data is not None:
        graded_errorbar(obs_data,
                        obs_sigma,
                        ms=data_ms,
                        ax=ts_ax,
                        color=c_data,
                        zorder=5, alpha=0.8)

    if true_curve is not None:
        ts_ax.plot(tsteps, true_curve(tsteps), ls='--', c=c_true,
                   label='true',
                   lw=lw_overplot)

    if t_forecast is not None:
        forecast_data = forecast_ensemble(ensemble, samples, t_forecast)
        ts_ax.axvline(t_forecast,
                      ls=ls_overplot,
                      lw=lw_overplot,
                      color=c_forecast,
                      alpha=alpha_forecast,
                      )
        ts_ax.axhline(np.mean(forecast_data),
                      ls=ls_overplot,
                      lw=lw_overplot,
                      color=c_forecast,
                      alpha=alpha_forecast,
                      )

        hist_ax = plt.subplot(gs[1])
        hist_ax.axhline(np.mean(forecast_data),
                        ls=ls_overplot,
                        lw=lw_overplot,
                        color=c_forecast,
                        alpha=alpha_forecast,
                        )
        hist_ax.hist(forecast_data, orientation='horizontal',
                     color=c_trace)
        _ = hist_ax.set_ylim(ts_ax.get_ylim())
        if true_curve:
            hist_ax.axhline(true_curve(t_forecast),
                            ls=ls_overplot,
                            c=c_true)
    plt.legend()