'''
This code contains statistics related functions

Written by Pablo Lemos(UCL)
pablo.lemos.18@ucl.ac.uk
Feb 2019
'''

import numpy as np
from scipy.special import erfinv, erfcinv

def get_nsigma(pte):
    '''Calculate the number of sigma for a given Probability to 
    Exceed (PTE)

    Parameters
    ----------
    pte: float
        Probability to Exceed
    
    Returns
    -------
    nsigma: float
        The corresponding number of sigma

    '''

    return np.sqrt(2)*erfcinv(pte)

def get_pte_from_samples(x, loglike_x, loglike_samples, weights = 1):
    '''Calculate the PTE for a given point its corresponding 
    log-likelihood, and the log-likelihood of samples

    Parameters
    ----------
    x: array(ndim)
        The point for which we are calculating the pte
    loglike_x: float
        The log-likelihood of x
    loglike_samples: array(nsamples)
        The log-likelihood of samples from the probability distribution
        function
    weights: array(nsamples)
        The sample weights. Defaults to 1, in each case all samples are
        equally weighted

    Returns
    -------
    pte: float
        The corresponding probability to exceed.
    '''
    # Number of samples
    nsamples = len(loglike_samples)
    
    # If there are no weights, everything has weight = 1
    if type(weights) is int: 
        weights = np.ones(nsamples)
    
    # Count how many samples have a higher likelihood than x
    total = 0
    for i in range(nsamples):
        if loglike_samples[i] > loglike_x:
            total += weights[i]

    # Get PTE
    pte=1-total/float(np.sum(weights))

    return pte
    