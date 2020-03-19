'''
This code calculates the `difference chain' from two given chains (MCMC 
or Nested Sampling). 


Written by Pablo Lemos(UCL)
pablo.lemos.18@ucl.ac.uk
Feb 2019
'''

import numpy as np

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

def generate_difference_chain(path_to_chain_A, path_to_chain_B):
    ''' Generate a difference chain from two existing chains. This 
    method does not calculate the likelihood, only the sample 
    weights 

    Parameters
    ----------
    path_to_chain_A, path_to_chain_B: str
        The path to each chain

    Returns
    -------
    diff_weights: array(nsamples)
        The weights of the new samples
    diff_samples: array(ndim,nsamples)
        The samples from the difference chain
    '''

    # Import the chains
    weights_A, loglike_A, X_A = load_chains(path_to_chain_A)
    weights_B, loglike_B, X_B = load_chains(path_to_chain_B)

    # Length of each chain
    len_A, len_B = len(weights_A), len(weights_B)

    # Number of parameters
    ndim = len(X_A[0])
    ndim = 5

    # Create difference chain
    diff_weights = np.empty([len_A*len_B])
    diff_loglike = np.empty([len_A*len_B])
    diff_samples = np.empty([len_A*len_B, ndim])

    # Index of new chain
    k=0

    # Iterate over all samples
    for i in range(len_A):
        for j in range(len_B):
            # Calculate new weight
            weight = weights_A[i]*weights_B[j]

            # Only add the sample if it has non-zero weight
            if weight > 1e-50:
                diff_weights[k] = weight
                diff_loglike[k] = loglike_A[i] + loglike_B[j]
                diff_samples[k] = X_A[i,:5] - X_B[j,:5]
                k+=1

    return diff_weights, diff_loglike, diff_samples

def select_samples(weights, loglike, X, nsamples):
    ''' Randomly select a number nsamples of samples from a chain

    Parameters
    ----------
    weights: array(nsamples_orig)
        The weights for each point in the original chain
    X: array(ndim,nsamples_orig)
        The samples in the original chain
    nsamples: int
        The number of samples to be kept. If it is larger than the 
        number of samples in the chain, all is kept.

    Returns:
    --------
    weights_short: array(nsamples)
        The randomly selected weights    
    X_short: array(ndim, nsamples)
        The randomly selected samples
    '''
    
    # The samples of the original chain
    nsamples_orig = len(weights)

    # Print a warning message and return if nsamples is larger than 
    # the number of samples in the original chain
    if nsamples > nsamples_orig: 
        print('''WARNING: In function select_samples, nsamples is larger
        than the number of samples in the original chain''')
        return weights, loglike, X 

    # Randomly select nsamples indices
    indices = np.random.choice(nsamples_orig, nsamples)

    weights_short = weights[indices]
    loglike_short = loglike[indices]
    X_short = X[indices]

    return weights_short, loglike_short, X_short

def main(path_to_chain_A, path_to_chain_B, nsamples):

    diff_weights, diff_loglike, diff_samples = generate_difference_chain(
        path_to_chain_A, path_to_chain_B)

    weights, loglike, samples = select_samples(diff_weights, diff_loglike, 
    diff_samples, nsamples)

    return samples, loglike, weights


    
