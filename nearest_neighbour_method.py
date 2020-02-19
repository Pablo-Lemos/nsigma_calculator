'''
This code calculates the tension as Probability To Exceed (PTE), for a given 
point and a chain (MCMC or nested sampling) using the nearest neighbour. 

Written by Pablo Lemos(UCL)
pablo.lemos.18@ucl.ac.uk
Feb 2019
'''

import numpy as np
from kde_method import load_chains
from stats import get_pte_from_samples

def find_min_max(X):
    ''' Finds the minimum and maximum value of each coordinate for a chain

    Parameters
    ----------
    X: array(ndim, nsamples)
        The samples

    Returns
    -------
    min_X, max_X: array(ndim)
        The minimum and maximum on each dimension
    '''

    # The number of dimensions
    ndim = len(X[0])

    # Create the empty arrays
    min_X, max_X = np.zeros(ndim), np.zeros(ndim)

    # Iterate over the number of dimensions
    for i in range(ndim):
        min_X[i] = min(X[:,i])
        max_X[i] = max(X[:,i])

    return min_X, max_X

def normalize_chain(X):
    ''' Changes coordinates such that all dimensions go from 0 to 1

    Parameters
    ----------
    X: array(ndim, nsamples)
        The original samples

    Returns
    -------
    X_norm: array(ndim, nsamples)
        The normalized samples
    min_X, max_X: array(ndim)
        The minimum and maximum on each dimension
    '''

    # Find the minimum and maximum value for each dimension
    min_X, max_X = find_min_max(X)

    # Normalize the chains
    X_norm = (X + min_X)/(max_X - min_X)
 
    return X_norm, min_X, max_X

def sample_distance(x1, x2):
    ''' Calculate the relative distance between two samples

    Parameters
    ----------
    x1, x2: array(ndim)
        The two samples
    
    Returns
    -------
    dist: float
        The relative distance between the samples
    '''

    dist = np.sum(np.sqrt((x1-x2)**2.))
    return dist

def find_nearest_neighbour(x0, X):
    ''' Find sample in a chain closest to given sample

    Parameters
    ----------
    x0: array(ndim)
        The sample whose nearest neighbour we want to find.
    X: array(ndim, nsamples)
        The samples from the chain

    Returns
    -------
    nearest_neighbour: array(ndim)
        The nearest neighbour from the chain samples
    min_index: int
        The index of the nearest neighbour
    '''
    
    # Number of samples
    nsamples = len(X)

    # Normalize chains
    X_norm, min_X, max_X = normalize_chain(X)
    x0 = (x0 + min_X)/(max_X - min_X)

    # Initial minimal distance and index
    min_dist, min_index= 1e300, 0

    # Iterate through all the samples
    for i in range(nsamples):
        dist = sample_distance(x0, X_norm[i])
        if dist < min_dist:
            min_dist = dist
            min_index = i

    # Save the (unnormalized) nearest neighbour
    nearest_neighbour = X[min_index]

    return nearest_neighbour, min_index

def get_pte_nearest_neighbour(x, path_to_chains):
    '''Calculate the point for a given point and a chain, using the 
    nearest neighbour method

    Parameters
    ----------
    x: array(ndim)
        The point for which we are calculating the pte
    path_to_chains: str
        The path to the chains

    Returns
    -------
    pte: float
        The corresponding probability to exceed.
    '''
    
    # Load chain
    weights, loglike, samples = load_chains(path_to_chains)

    # Find nearest neighbour
    nearest_neighbour, index = find_nearest_neighbour(x, samples)
    print(nearest_neighbour)

    # The log-likelihood for x
    loglike_x = loglike[index]
    print(loglike_x)

    # Get PTE
    pte = get_pte_from_samples(samples, loglike_x, loglike, weights)

    return pte