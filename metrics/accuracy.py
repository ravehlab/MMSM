import numpy as np
from msmtools.analysis.dense.stationary_vector import stationary_distribution
from .util import *


def boltzmann_test(hmsm, level, discretizer, energy, kT, n_samples=15):
    """boltzmann_test.
    Get the KL-divergence between the stationary distribution on states on a given level and the 
    Boltzmann distribution on those states, conditioned on being in one of the states covered by
    the HMSM.

    Parameters
    ----------
    hmsm :
        hmsm
    level :
        level
    discretizer :
        discretizer
    energy : callable
        energy function R^d -> R
    kT : float
        kT
    n_samples : int
        number of samples used to estimate the free energy of each microstate

    Returns
    -------"""
    pi, ids = induced_distribution(hmsm, 0, level)

    discrete_free_energy = np.ndarray(pi.shape)
    for i, id in enumerate(ids):
        discrete_free_energy[i] = \
                np.mean([energy(discretizer.sample_from(id)) for i in range(n_samples)])
    boltzmann = np.exp(-discrete_free_energy/kT)
    boltzmann *= (1/np.sum(boltzmann))
    return D_KL(boltzmann, pi)

    


def discretization_error(hmsm, descendant_level, parent_level):
    """discretization_error.
    Get the KL-divergence between the stationary distribution of the MSM on level descendant_level,
    and the distribution induced by the stationary distribution of the MSM on level parent_level.
    
    The induced stationary distribution on an MSM :math:`M_i` of level i, from an MSM :math:`M_j` on level :math:`j>i` is
    defined as follows:
    If :math:`j=i+1` then p(x) for a state :math:`x \in M_i` with parent state :math:`y in M_j'`, is defined as 
    .. math::
        p_{M_j}(y)\cdot p_{y}(x)
    Where :math:`p_{M_j}, p_{y}` are the stationary distribution of :math:`M_j, y` respectively.

    If j>1, then p(x) for a state :math:`x \in M_i` with parent state :math:`y \in M_{i+1}`, is defined as:
        p_{M_{i+1}}(y)\cdot p_{y}(x)
    Where :math:`p_{M_{i+1}}` is the induced stationary distribution of :math:`M_j` on :math:`M_{i+1}`

    Parameters
    ----------
    hmsm :
        hmsm
    descendant_level :
        descendant_level
    parent_level :
        parent_level

    Returns
    -------
    KL_divergence : float
        The Kullback-Leibler divergence between the stationary distribution and induced stationary
        distribution
    """
    if descendant_level==0:
        pi_dict = hmsm.get_full_stationary_distribution()
        # pi is an np array of the probabilities, sorted by their ids:
        pi = np.array(list(pi_dict.values()))  #just the distribution
        pi = pi[np.argsort(list(pi_dict.keys()))] # sort by the ids
    else:
        T, ids = hmsm.get_level_T(descendant_level, 1)
        pi = stationary_distribution(T)
        pi = pi[np.argsort(ids)]

    induced_pi, ids = induced_distribution(hmsm, descendant_level, parent_level)
    induced_pi = induced_pi[np.argsort(ids)]

    return D_KL(pi, induced_pi)

def MHT_uncertainty(tree, nsamples=100, sink=0):
    T_samples, ids = tree.sample_full_T(nsamples)
    mht = mean_hitting_time(T_samples, sink)
    return ids, np.var(mht, axis=0)

