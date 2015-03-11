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
    rngs=[pd.Index([start])]
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

def build_covariance_matrix(variances, correlations):
    """
    Builds a covariance matrix from variances and correlations.

    Args:
        variances (dict): Dict mapping parameter name to variance.
            If column/row ordering of the covariance matrix is important, this
            should be an instance of :py:class:`collections.OrderedDict`.
        correlations (dict): Mapping of tuples to correlations, e.g.
            ``{(param1,param2):0.5}``. Default correlation for unspecified
            pairs is zero.
    Returns:
        (DataFrame): Covariance matrix.
    """
    cov=pd.DataFrame(
        data=np.diag(np.array(variances.values(),dtype=np.float)),
        columns=variances.keys(),
        index=variances.keys())
    for param_pair, pair_corr in correlations.items():
        p1,p2=param_pair
        pair_cov = pair_corr * np.sqrt(variances[p1]*variances[p2])
        cov.loc[p1,p2] = cov.loc[p2,p1] = pair_cov
    return cov