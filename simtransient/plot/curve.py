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



    ax.errorbar(obs_data.index, obs_data,
                label=label,
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