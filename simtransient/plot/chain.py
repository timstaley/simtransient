"""
Plots for displaying sample chains
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def param_walkers(rawchain,
                  chainstats,
                  param_idx,
                  param_name=None,
                  axes=None):
    if param_name is None:
        param_name = ''
    if axes is None:
        fig, axes = plt.subplots(nrows=2, ncols=1)
    ax0, ax1 = axes
    cs = chainstats

    ndim = rawchain.shape[-1]
    alldata = rawchain[:, :, param_idx].ravel()
    dmin, dmax = np.percentile(alldata, (0.01, 99.99))
    drange = dmax - dmin
    ymin = dmin - drange / 10
    ymax = dmax + drange / 10

    ax0.set_title('{} (Raw)'.format(param_name))
    ax0.set_ylabel(param_name)
    # for walker in rawchain[:, :, param_idx]:
    for walkerdata in rawchain:
        ax0.plot(walkerdata[:, param_idx])
    ax0.axvline(cs.nburn, ls=':', color='k')
    ax0.set_ylim(ymin, ymax)

    ax1.set_title('{} (Thinned)'.format(param_name))
    ax1.set_ylabel(param_name)
    # plt.xlabel("Sample")
    for walkerdata in rawchain:
        ax1.plot(walkerdata[::cs.acorr, param_idx])
    ax1.axvline(cs.nburn / cs.acorr, ls=':', color='k')
    ax1.get_yaxis().set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax1.set_ylim(ymin, ymax)


def all_walkers(rawchain,
                chainstats,
                param_names,
                axes=None):
    ndim = len(param_names)
    if axes is None:
        fig, axes = plt.subplots(nrows=ndim, ncols=2)

    for idx, name in enumerate(param_names):
        ax_pair = axes[idx, :]
        param_walkers(rawchain,
                      chainstats,
                      idx,
                      name,
                      axes=ax_pair
                      )
    plt.tight_layout()


def param_hist(rawchain,
               chainstats,
               param_idx,
               param_name=None,
               nbins=15,
               axes=None,
               ):
    if param_name is None:
        param_name = ''
    if axes is None:
        axes = plt.gca()
    cs = chainstats

    axes.set_xlabel(param_name)
    axes.set_ylabel("Frequency")

    burned_data = rawchain[:, cs.nburn:, param_idx].ravel()
    xmin, xmax = np.percentile(burned_data, (0.01, 99.99))
    plt_idx = np.logical_and(burned_data > xmin, burned_data < xmax)
    plot_data = burned_data[plt_idx]
    plot_data_thinned = plot_data[::cs.acorr]

    _, bin_edges, _ = axes.hist(plot_data,
                                normed=True,
                                bins=nbins,
                                alpha=0.8,
                                label='Raw')
    axes.hist(plot_data_thinned,
              bins=bin_edges,
              normed=True, alpha=0.5,
              label='Thinned')
    # axes.set_xlim(xmin,xmax)
    axes.legend(loc='best')


def all_hists(rawchain,
              chainstats,
              param_names,
              nbins=15,
              axes=None):
    ndim = len(param_names)
    if axes is None:
        nrows = int(np.ceil(ndim / 2.0))
        fig, axes = plt.subplots(nrows=nrows, ncols=2)

    for idx, name in enumerate(param_names):
        ax = axes.ravel()[idx]
        param_hist(rawchain,
                   chainstats,
                   idx,
                   param_name=None,
                   nbins=nbins,
                   axes=ax
                   )
        ax.set_title(name)
    plt.tight_layout()

