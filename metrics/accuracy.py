import numpy as np
from msmtools.analysis.dense.stationary_vector import stationary_distribution
from .util import D_KL, induced_distribution, mean_hitting_time, chi_squared_distance


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
    var = np.var(mht, axis=0)
    mean = np.mean(mht, axis=0)
    return ids, np.nan_to_num(var/mean, copy=False)

def kinetic_error(base_T, base_ids, est_T, est_ids) -> dict:
    """kinetic_error.

    Parameters
    ----------
    base_T : np.ndarray of shape (n,n)
        Ground truth transition matrix
    base_ids : np.ndarray of shape (n), dtype=int
        The ids of the states represented by base_T - i.e. base_T[i,j] is the transition proability
        from state base_ids[i] to base_ids[j]
    est_T : np.ndarray of shape (k,k)
        Estimated transition matrix.
    est_ids : np.ndarray of shape (k), dtype=int
        The ids of the states represented by est_T - i.e. est_T[i,j] is the transition proability
        from state est_ids[i] to est_ids[j]
        The states est_ids must be a subset of the states base_ids, otherwise a ValueError will be
        raised.

    Returns
    -------
    errors : dict[int, float]
        A dictionary such that errors[id] is the Kullback-Liebler divergence between the transition
        probabilities of id in base_T to the transition probabilities of id in est_T.
    """
    n = base_T.shape[0]
    k = est_T.shape[0]
    # map from indices of est_T/ids to corresponding indices of base_T/ids:
    est_inx_2_base_inx = np.array([np.argmax(base_ids==state) for state in est_ids], dtype=int)

    def est_row_2_base_row(i):
        """Reshape the row est_T[i] such that the indices are the same as base_T,
           in other words: est_T[est_row_2_base_row(i)] is an estimate of
           base_T[est_inx_2_base_inx[i]]
        """
        row = np.zeros(n)
        for j in range(k):
            row[est_inx_2_base_inx[j]] = est_T[i,j]
        return row

    # populate the 'error' dictionary
    error = dict.fromkeys(base_ids, np.inf)
    for i, state in enumerate(est_ids):
        error[state] = chi_squared_distance(
                                            est_row_2_base_row(i), 
                                            base_T[est_inx_2_base_inx[i]]
                                            )

    return error


