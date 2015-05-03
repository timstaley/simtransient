"""
Subroutines for various aspects of Bayesian calculations, etc.
"""
from __future__ import absolute_import
import numpy as np
import pandas as pd
import statsmodels.api as sm

from itertools import combinations_with_replacement
from collections import OrderedDict
from scipy.interpolate import interp1d

def generate_conditional_pmf_grids(model_feature_pdfs,
                                  feature_min,
                                  feature_max,
                                  feature_delta,
                                  feature_name=None):
    """
    Sample feature PDFs (at a single epoch) on a grid for numerical integration.

    We prepare a set of discrete 'probability mass function' samplings for the
    feature PDFs of each class, i.e. P[D_f|e,l_i]

    The samplings are on a common grid, so that the PMFs of different classes
    may be combined to calculate utility functions.
    We also utilise the priors for each class to calculate the total 'evidence'
    at each grid position, i.e. P[D_f|e]

    Args:
        model_priors (Series): Prior probability of each class
        model_feature_pdfs (Series): PDF function for the feature in question of
            each class.
        feature_name (str): Used for naming the Index of resulting DataFrame.
        feature_minmax (tuple of float): Defines range of feature grid.
        feature_delta (float): Defines stepsize of feature grid.
    Returns:
        pmf_grids, weighted_pmf_grids, evidence (DataFrame,DataFrame,Series):
            'pmf_grids' are PMF sampling series for each class,
            i.e.  P[D_f|e,t_i] sampled onto a fine discrete grid.
            'weighted_pmf_grids' are as above but weighted by priors, i.e.
            P[D_f|e,t_i]P[t_i]

            Results are indexed by data-position, one column per class.
    """
    feature_domain_grid = np.arange(feature_min,
                                    feature_max + feature_delta,
                                    feature_delta)
    pmf_grids = pd.DataFrame(
        index=pd.Index(feature_domain_grid, name=feature_name))
    for mod, pdf in model_feature_pdfs.iteritems():
        pmf_grids[mod] = pd.Series(pdf(feature_domain_grid) * feature_delta,
                                   pmf_grids.index, name=mod)
    pmf_grids = pmf_grids / pmf_grids.sum()
    # weighted_pmf_grids = pmf_grids.mul(model_priors.astype(float))
    rowsum = pmf_grids.T.sum()
    #Drop any points where all the PMFs vanish
    valid_index = rowsum > 0.
    pmf_grids = pmf_grids[valid_index]
    return pmf_grids


def autogenerate_conditional_pmf_grids(modelruns,
                                       t_forecast,
                                       n_feature_samples,
                                       feature_name=None):
    named_kdes = pd.Series()
    for mr in modelruns:
        forecast_data = mr.compute_forecast_data(t_forecast)
        kde = mr.get_kde(forecast_data)
        named_kdes[mr.ensemble.classname]=kde
    model_pdfs = named_kdes.map(lambda kde:kde.evaluate)
    model_feature_mins = named_kdes.map(lambda kde:kde.support.min())
    model_feature_maxes = named_kdes.map(lambda kde:kde.support.max())
    fmin = model_feature_mins.min()
    fmax = model_feature_maxes.max()
    delta = (fmax - fmin)/float(n_feature_samples)
    return generate_conditional_pmf_grids(model_pdfs,
                                         fmin,
                                         fmax,
                                         delta,
                                         feature_name)



def compute_confusion_matrix(model_priors, pmfs):
    """
    Args:
        model_priors (Series): Prior probability of each class
        pmfs (DataFrame): Conditional probability mass function, for each class
            (A sampling of the PDFs on a shared grid.)
    """

    # 'sum_over_classes' is the sum at each data-location of the weighted_pmf,
    # i.e.
    # \sum_k(P[D_f|e,t_k]P[t_k])
    sum_over_classes = pmfs.mul(model_priors.astype(float)).T.sum()
    #Evidence may go to zero if some classes are already ruled out
    valid_index = sum_over_classes > 0.0
    sum_over_classes = sum_over_classes[valid_index]
    valid_pmfs = pmfs[valid_index]

    confusion = pd.DataFrame(
        [pd.Series(index=model_priors.index, name=mod)
         for mod in model_priors.index],
        index=pd.Index(model_priors.index, name="True class"), )

    pairings = combinations_with_replacement(model_priors.index, r=2)
    for true_class, label_class in pairings:
        #NB element-wise mult/div is simple for series:
        integral = (
            valid_pmfs[true_class] * valid_pmfs[label_class] / sum_over_classes).sum()
        confusion.loc[true_class, label_class] = model_priors[label_class] * integral
        #When label / true are reversed, the calcs are same but for prior:
        if true_class != label_class:
            label_class, true_class = true_class, label_class
            confusion.loc[true_class, label_class] = model_priors[label_class] * integral

    return confusion


def compute_information_score(model_priors, pmfs):
    """
    Calculates the expected utility using the information content metric.

    Args:
        model_priors (Series): Prior probability of each class
        pmfs (DataFrame): Conditional probability mass function, for each class
            (A sampling of the PDFs on a shared grid.)

    """
    # P[D_f | e,t_i]P[t_i]
    weighted_pmfs = pmfs.mul(model_priors.astype(float))
    # \sum_k(P[D_f|e,t_k]P[t_k])
    sum_over_classes = weighted_pmfs.T.sum()

    valid_index = sum_over_classes > 0.0
    sum_over_classes = sum_over_classes[valid_index]
    weighted_pmfs = weighted_pmfs[valid_index]

    # P[D_f | e,t_i]P[t_i]
    #        / P[D_f|e]
    normed_weighted_pmfs = weighted_pmfs.div(sum_over_classes, axis='index')
    log_nw_pmfs = np.log(normed_weighted_pmfs)
    # Wherever PMF==0, log(PMF)=-inf.
    # Now, 0 * -inf = nan
    # which is subsequently treated as 0 in the sum anway,
    # so don't strictly need to do anything.
    # But we replace -inf with 0-values anyway, for clarity / future proofing
    log_nw_pmfs[log_nw_pmfs == -float('inf')] = 0.

    #Utility for each possible data-outcome; series indexed by feature-value
    utility_series = (normed_weighted_pmfs * log_nw_pmfs).T.sum()

    #expected utility:
    # multiply utility at each data-outcome by P[t_j]P[D|e,t_j].
    # Then integrate (sum over feature-space-grid), and sum over classes
    eu = weighted_pmfs.mul(utility_series, axis='index').sum().sum()
    return eu
