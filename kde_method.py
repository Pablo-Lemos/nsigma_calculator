'''
This code calculates the tension as Probability To Exceed (PTE), for a given 
point and a chain (MCMC or nested sampling) using Kernel Density Estimation
(KDE). 

Written by Pablo Lemos(UCL)
pablo.lemos.18@ucl.ac.uk
Feb 2019
'''

import numpy as np
from stats import get_pte_from_samples
from scipy.stats import gaussian_kde

def load_chains(path_to_chains):
    '''Load MCMC or Nested sampling chains

    Parameters
    ----------
    path_to_chains: str
        The path to the chains
    
    Returns
    -------
    weights: array(nsamples)
        The weights for each point
    loglike: array(nsamples)
        The log-likelihood for each point
    X: array(ndim,nsamples)
        The samples
    '''

    # Add .txt at the end of the string if not present
    if path_to_chains[-4:] is not '.txt':
        path_to_chains += '.txt'
    
    # Load the chains
    samples = np.loadtxt(path_to_chains)
    
    # Assign the different columns
    weights = samples[:,0]
    chi2 = samples[:,1]
    X = samples[:,2:]

    # Turn chi2 into loglike
    loglike = -0.5*chi2

    return weights, loglike, X

def fit_kde(X, weights = 1, kernel='gaussian', bandwidth=0.2, rtol = 1e-8):
    '''Fit the samples using KDE

    Parameters
    ----------
    weights: array(nsamples)
        The weights for each point
    X: array(ndim,nsamples)
        The samples
    kernel: string
        The KDE kernel to be used, defaults to 'gaussian'
    bandwidth: float
        The KDE bandwidth to be used, defaults to 0.1
    rtol: float
        Relative tolerance of the KDE fitting. Defaults to 1e-8

    Returns
    ----------
    kde: model
        The fitted KDE
    '''

    # Do we have weights
    if type(weights) is int:
        # Fit the data (no weights)
        kde = gaussian_kde(X.T, bw_method='silverman')
    else:
        # Fit the data (no weights)
        kde = gaussian_kde(X.T, bw_method='silverman', weights = weights.T)

    return kde

def get_kde_loglike(X, kde):
    '''Get an estimate of the log likelihood using KDE

    Parameters
    ----------
    X: array(ndim,nsamples)
        The samples
    kde: model
        A KDE fitted to the samples
    
    Returns
    ----------
    loglike: array(nsamples)
        The log-likelihood for each point estimated from a KDE
    '''

    # Get the log likelihood
    loglike = kde.logpdf(X.T)
    
    return loglike

def get_pte_KDE(x, path_to_chains):
    '''Calculate the point for a given point and a chain, using KDE

    Parameters
    ----------
    x: array(ndim)
        The point for which we are calculating the pte
    path_to_chains: str
        The path to the chains
    kernel: string
        The KDE kernel to be used, defaults to 'gaussian'
    bandwidth: float
        The KDE bandwidth to be used, defaults to 0.2
    rtol: float
        Relative tolerance of the KDE fitting. Defaults to 1e-8

    Returns
    -------
    pte: float
        The corresponding probability to exceed.
    '''

    # Load chain
    weights, loglike_samples, X = load_chains(path_to_chains)

    # Perform KDE fitting
    kde = fit_kde(X, weights)

    # Get KDE log-likelihood
    loglike_kde = get_kde_loglike(X, kde)

    # The log-likelihood for x
    loglike_x = get_kde_loglike([x], kde)

    # Get PTE
    pte = get_pte_from_samples(x, loglike_x, loglike_kde, weights = weights)

    return pte
