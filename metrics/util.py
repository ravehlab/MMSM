import numpy as np
from scipy.special import rel_entr
from msmtools.analysis.dense.stationary_vector import stationary_distribution


def induced_distribution(hmsm, descendant_level, parent_level):
    if (descendant_level > parent_level) or (parent_level > hmsm.height) or (descendant_level<0):
        raise ValueError(f"Recieved invalid parent and descendent levels for HMSM of height {hmsm.height}: parent: {parent_level}, descendant: {descendant_level}.\n")

    induced_level = parent_level
    T, ids = hmsm.get_level_T(induced_level, 1)
    pi = stationary_distribution(T)
    while (induced_level > descendant_level):
        induced_level -= 1
        if induced_level==0:
            n = len(hmsm._microstate_parents)
        else:
            n = len(hmsm._levels[induced_level])
        induced_level_ids = []
        induced_level_pi = np.zeros(n)
        i = 0
        for vertex_id, p in zip(ids, pi):
            vertex = hmsm.vertices[vertex_id]
            children = vertex.children
            induced_level_ids += children
            induced_level_pi[i:i+len(children)] = p*vertex.get_local_stationary_distribution()
            i += len(children)
        ids, pi = induced_level_ids, induced_level_pi
        assert i==n
        assert np.isclose(np.sum(pi), 1.0) , np.sum(pi)
    return pi, ids

def D_KL(p, q):
    assert p.shape==q.shape
    assert p.ndim==q.ndim==1
    return np.sum(np.fromiter(map(rel_entr, p, q), dtype=np.double))

def mean_hitting_time(T, i):
    """mean_hitting_time.
    Gets the mean hitting time from all nodes to node i.

    Parameters
    ----------
    T : ndarray
        Transition matrix of shape (_,n,n)
    i : int
        Index of the node

    Returns
    -------
    x : ndarray
        An array of shape (_,n), such that x[_,j] is the mean hitting time of i from j.
    """
    n = T.shape[1]
    T_i = T - np.eye(n)
    if T.ndim == 3:
        nsamples = T.shape[0]
        T_i[:,i] = np.zeros(n)
        T_i[:,i,i] = 1
        b = -np.ones((nsamples, n))
        b[:,i] = 0
    else:
        T_i[i] = np.zeros(n)
        T_i[i,i] = 1
        b = -np.ones(n)
        b[i] = 0
    return np.linalg.solve(T_i, b)
