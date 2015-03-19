"""
Plots for displaying lightcurves (and observation data)
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def scatter_graded_errorbars(obs_data,
                             obs_sigma,
                             color,
                             base_elw=1.5,
                             ax=None,
                             **kwargs):
    if ax is None:
        ax = plt.gca()

    ax.errorbar(obs_data.index, obs_data,
                c=color,
                linewidth=0,
                elinewidth=0,
                ms=16,
                marker='.',
                **kwargs
                )
    ax.errorbar(obs_data.index, obs_data, yerr=1 * obs_sigma,
                c=color,
                linewidth=0,
                elinewidth=base_elw*3,
                ms=16,
                marker='.',
                **kwargs
                )
    ax.errorbar(obs_data.index, obs_data, yerr=2 * obs_sigma,
                c=color,
                linewidth=0,
                elinewidth=base_elw,
                ms=0,
                marker='',
                **kwargs
                )
    # ax.scatter(obs_data.index, obs_data, s=50, c=[color])