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