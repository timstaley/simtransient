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
                    label=None,
                    color='g',
                    base_elw=1.5,
                    ms=16,
                    ax=None,
                    **kwargs):
    if ax is None:
        ax = plt.gca()

    first_plot_kwargs = kwargs.copy()
    if label is not None:
        first_plot_kwargs['label']=label

    ax.errorbar(obs_data.index, obs_data,
                ecolor=color,
                mfc=color,
                linewidth=0,
                elinewidth=0,
                ms=ms,
                marker='.',
                **first_plot_kwargs
                )
    if obs_sigma is not None:
        ax.errorbar(obs_data.index, obs_data, yerr=1 * obs_sigma,
                    ecolor=color,
                    mfc=color,
                    linewidth=0,
                    elinewidth=base_elw * 3,
                    ms=0,
                    marker='.',
                    **kwargs
                    )
        ax.errorbar(obs_data.index, obs_data, yerr=2 * obs_sigma,
                    ecolor=color,
                    mfc=color,
                    linewidth=0,
                    elinewidth=base_elw,
                    ms=0,
                    marker='',
                    **kwargs
                    )
        # ax.scatter(obs_data.index, obs_data, s=50, c=[color])