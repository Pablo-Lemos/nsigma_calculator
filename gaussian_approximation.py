'''
This code calculates the tension as Probability To Exceed (PTE), for a given 
point and a Gaussian likelihood with given mean and covariance matrix. 

Written by Pablo Lemos(UCL)
pablo.lemos.18@ucl.ac.uk
Feb 2019
'''

import numpy as np
from scipy.stats import norm, multivariate_normal
from stats import get_pte_from_samples

def generate_samples(mean, cov, nsamples):
    '''Generate samples from a Gaussian distribution, and calculate their 
    corresponding log likelihood.

    Parameters
    ----------
    mean: array(ndim)
        Mean of the Gaussian distribution
    cov: array(ndim, ndim)
        Covariance matrix of the Gaussian distribution
    nsamples: int
        The number of samples to generate

    Returns
    -------
    X: array(ndim, nsamples)
        An array of samples from the Gaussian distribution
    loglike: array(nsamples)
        The corresponding log likelihood.
    '''     
    
    # One dimensional case
    if ((type(mean) is float) or (type(mean) is int)): 
        # Generate samples
        X = norm.rvs(loc=mean, scale=cov, size = nsamples)
        # Calculate log-likelihood
        loglike = norm.logpdf(X, loc=mean, scale=cov)

    # Multidimensional case
    else:
        # Generate samples
        X = multivariate_normal.rvs(mean=mean, cov=cov, size = nsamples)
        # Calculate log-likelihood
        loglike = multivariate_normal.logpdf(X, mean=mean, cov=cov)

    return X, loglike

def get_pte_from_gaussian(x, mean, cov, nsamples):
    '''Calculate the PTE for a given point and Gaussian distribution

    Parameters
    ----------
    x: array(ndim)
        The point for which we are calculating the pte
    mean: array(ndim)
        Mean of the Gaussian distribution
    cov: array(ndim, ndim)
        Covariance matrix of the Gaussian distribution
    nsamples: int
        The number of samples to generate

    Returns
    -------
    pte: float
        The corresponding probability to exceed.
    '''
    
    #Generate samples for Gaussian distribution
    y, loglike_samples = generate_samples(mean, cov, nsamples)

    # The log-likelihood for x
    loglike_x = multivariate_normal.logpdf(x, mean=mean, cov=cov)

    # Get PTE
    pte = get_pte_from_samples(x, loglike_x, loglike_samples)
    
    return pte
    

    