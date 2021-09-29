"""Partition using Gibbs metastable clustering"""

# Author: Kessem Clein <kessem.clein@mail.huji.ac.il>
import warnings
import numpy as np
import numba
from HMSM.util.linalg import normalize_rows, _assert_stochastic
from .util import get_size_or_timescale_split_condition, spectral
from . import MetastablePartition

__all__ = ['GibbsPartition']

class GibbsPartition(MetastablePartition):
    """GibbsPartition.
    """


    def __init__(self, max_k_method='default', transition_parameter=0.5, \
                 max_size=2048, max_timescale=32, mle=False):
        self.max_k_method = max_k_method
        self.transition_parameter = transition_parameter
        self._split_condition = get_size_or_timescale_split_condition(max_size, max_timescale)
        self._mle = mle

    def _get_max_k(self, n):
        if self.max_k_method in ('default', '2_sqrt'):
            max_k = 2*int(np.sqrt(n))
            return min(max_k, n-1)
        #TODO max_k should depend on number of eigenvalues with timescale greater than 2*Tau,
        #     or something similar
        raise NotImplementedError(f"max_k_method {self.max_k_method} not implemented")

    def check_split_condition(self, vertex):
        return self._split_condition(vertex)

    def get_metastable_partition(self, vertex):
        """Get a partition of a vertex
        """
        n = vertex.n
        T = normalize_rows(vertex.T[:n, :n], norm=1)
        partition = _gibbs_metastable_clustering(T, 
                                                 self.transition_parameter,
                                                 maximum_likelihood=self._mle)
        taus = [vertex.tau]*len(partition) #TODO get method for calculating taus in init
        return partition, taus

def _gibbs_metastable_clustering(T, transition_parameter, init='singletons', manual_init=None,
                                max_k=None,
                                maximum_likelihood=True, min_iter=50, max_iter=500,
                                mle_fraction=0.1):
    n = T.shape[0]
    if init=='spectral':
        #TODO decument this function. max_k needs to be provided if init=='spectral'
        initial_clustering = spectral(T, max_k)
    elif init=='manual':
        initial_clustering = manual_init
    else:
        initial_clustering = np.arange(n)

    n_iter = min([n, max_iter])
    n_iter = max([n_iter, min_iter])
    gibbs_clusters = _rw_gibbs(T, initial_clustering, n_iter, \
                               maximum_likelihood, transition_parameter)
    if maximum_likelihood:
        # the last iteration _is_ the MLE, so we only need one sample
        clustering = _get_MLE_clusters(gibbs_clusters, n_samples=1)
    else:
        clustering = _get_MLE_clusters(gibbs_clusters, mle_fraction)

    clustering_as_partition_of_indices = [np.where(clustering==cluster_index)[0]\
                                            for cluster_index in np.unique(clustering)]
    return clustering_as_partition_of_indices

def _rw_gibbs(T, init, n_iter=10, maximum_likelihood=True, transition_parameter=0.75):
    n = T.shape[0]
    k = len(init)
    clusters = np.zeros((n_iter+1,n), dtype=int)
    clusters[0] = init

    # RANDOM WALK:
    for it in range(n_iter):
        # next_step[i,j] gives the probability of the i'th vertex landing on a vertex with color
        # j, given the coloring from the previous iteration.
        next_step = _get_next_step(clusters[it], T, n)
        _assert_stochastic(next_step)
        for vertex in range(n):
            # now choose a coloring from the distribution defined above for the next iteration
            if maximum_likelihood:
                if np.max(next_step[vertex]) < transition_parameter:
                    # make no change to this vertex
                    clusters[it+1][vertex] = clusters[it][vertex]
                    continue
                color = np.argmax(next_step[vertex])
            else:
                color = np.random.choice(k, p=next_step[vertex])
            clusters[it+1][vertex] = color
        if maximum_likelihood and np.all(clusters[it+1]==clusters[it]):
            return clusters[:it+2]

    if maximum_likelihood:
        warnings.warn("gibbs metastable clustering didn't converge, try increasing max_iter\
                            or changing split criteria") 
    return clusters

@numba.njit
def _get_next_step(clustering, T, n):
    next_step_probabilities = np.zeros((n,n))
    for row in range(n):
        for neighbor, p in enumerate(T[row]):
            next_step_probabilities[row][clustering[neighbor]] += p
    return next_step_probabilities

def _get_MLE_clusters(clusters, fraction=0.1, n_samples=None):
    # clusters is an array of shape (n_iter, n) s.t. clusters[m,i] is the color of i in iteration m
    if n_samples is None:
        n_iter = clusters.shape[0]
        n_samples = int(np.ceil(n_iter * fraction))
    n = clusters.shape[1]
    assignment = np.ndarray(n, dtype=int)
    for i in range(n):
        states, counts = np.unique(clusters[-n_samples:, i], return_counts=True)
        frequency = counts/sum(counts)
        assignment[i] = states[np.argmax(frequency)]
    return assignment
