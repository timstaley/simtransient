import types
from collections import Iterable
import numpy as np
import pandas as pd


def listify(x):
    """Ensure x is iterable; if not then enclose it in a list and return it."""
    if isinstance(x, types.StringTypes):
        return [x]
    elif isinstance(x, Iterable):
        return x
    else:
        return [x]


def multifreq_daterange(start, *timedelta_freq_tuples):
    """
    Convenience wrapper around pandas.date_range, appends multiple
    date ranges with specified timedelta periods and frequency spacings.
    """
    rngs = [pd.Index([start])]
    for tdelta, freq in timedelta_freq_tuples:
        prev_end = rngs[-1][-1]
        rngs.append(pd.date_range(prev_end, prev_end + tdelta, freq=freq))
    epochs = sum(rngs)
    return epochs


def convert_to_timedeltas(epochs, epoch0):
    """
    Convert an epochs to timedeltas (in seconds), relative to epoch0.

    Args:
        epochs: Iterable containing datetimes, e.g. output from pandas.date_range.
        epoch0: Reference date-time.

    Returns:
        Series; timedeltas in seconds, indexed by epoch.
    """
    offsets = pd.Series(
        (pd.Series(epochs, index=epochs) - pd.Series(epoch0, index=epochs)
         ).astype('timedelta64[s]'),
        name='t_offset')
    return offsets


def mahalanobis_sq(icov, vecs):
    """
    Calculates the Mahalanobis distance for many vectors at once.


    Args:
        icov: Inverse-covariance matrix
        vecs: A nested array containing the mean-subtracted vector positions,
            e.g. something of the form:

                np.asarray([vec1, vec2, ...])

            where vec1 etc are 1-dim ndarrays.

    Returns:
        (1-d ndarray): Array containing the square of the Mahalanobis
            distance for each vector.

    """
    return np.sum(vecs.T * np.dot(icov, vecs.T), axis=0)


def build_covariance_matrix(sigmas, correlations):
    """
    Builds a covariance matrix from sigmas and correlations.

    Args:
        sigmas (pandas.Series): Series mapping parameter name to std. dev.
        correlations (dict): Mapping of tuples to correlations, e.g.
            ``{(param1,param2):0.5}``. Default correlation for unspecified
            pairs is zero.
    Returns:
        (DataFrame): Covariance matrix.
    """
    cov = pd.DataFrame(index=sigmas.index,
                       columns=sigmas.index,
                       data=np.diag(sigmas ** 2),
                       dtype=np.float
                       )
    for param_pair, pair_corr in correlations.items():
        p1, p2 = param_pair
        pair_cov = pair_corr * sigmas[p1] * sigmas[p2]
        cov.loc[p1, p2] = cov.loc[p2, p1] = pair_cov
    return cov


def subsample_curves(ensemble, sample, tsteps, size=100, fixed_pars=None):
    size=min(size, len(sample))
    if fixed_pars is None:
        fixed_pars={}

    subsamples = sample[np.random.choice(len(sample), size=size, replace=False)]
    # Have tried allocated an empty 2d array, but this is as fast and simpler:
    ss_curves = np.array([ensemble.evaluate(tsteps, *theta, **fixed_pars)
                          for theta in subsamples])
    return ss_curves


def forecast_ensemble(ensemble, sample, t_forecast):
    return np.fromiter(
        (ensemble.evaluate(t_forecast, *theta) for theta in sample),
        dtype=np.float)
