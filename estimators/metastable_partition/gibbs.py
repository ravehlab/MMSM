"""Partition using Gibbs metastable clustering"""

# Author: Kessem Clein <kessem.clein@mail.huji.ac.il>

import numpy as np
from HMSM.util.linalg import normalize_rows
from .util import get_size_or_timescale_split_condition, spectral
from . import MetastablePartition

__all__ = ['GibbsPartition']

class GibbsPartition(MetastablePartition):

    def __init__(self, max_k_method='default', transition_parameter=0.75, \
                 max_size=2048, max_timescale=64):
        self.max_k_method = max_k_method
        self.transition_parameter = transition_parameter
        self._split_condition = get_size_or_timescale_split_condition(max_size, max_timescale)

    def _get_max_k(self, n):
        if self.max_k_method in ('default', '2_sqrt'):
            return 2*int(np.sqrt(n))
        #TODO max_k should depend on number of eigenvalues with timescale greater than 2*Tau,
        #     or something similar
        raise NotImplementedError(f"max_k_method {self.max_k_method} not implemented")

    def check_split_condition(self, hmsm):
        return self._split_condition(hmsm)

    def get_metastable_partition(self, hmsm):
        """Get a partition of a vertex, assuming the parent's time resolution tau is twice that of
        this vertex.
        """
        n = hmsm.n
        T = hmsm.T[:n, :n]
        tau = hmsm.tau
        max_k = min(self._get_max_k(n), n-1)
        #partition = _gibbs_metastable_clustering(T, 2*tau, max_k, self.transition_parameter)
        partition = _gibbs_metastable_clustering(T, 2*tau, max_k, self.transition_parameter, maximum_likelihood=False)
        taus = [tau]*len(partition) #TODO get method for calculating taus in init
        return partition, taus

def _gibbs_metastable_clustering(T, tau, max_k, transition_parameter, init='spectral', manual_init=None, \
                                maximum_likelihood=True, min_iter=50, max_iter=500, mle_fraction=0.1):
    if init=='spectral':
        initial_clustering = spectral(T, max_k)
    elif init=='manual':
        initial_clustering = manual_init
    else:
        initial_clustering=None

    n = T.shape[0]
    n_iter = min([n, max_iter])
    n_iter = max([n_iter, min_iter])
    gibbs_clusters = _rw_gibbs(T, max_k, tau, n_iter, initial_clustering, \
                               maximum_likelihood, transition_parameter)
    if maximum_likelihood:
        # the last iteration _is_ the MLE, so we only need one sample
        clustering = _get_MLE_clusters(gibbs_clusters, n_samples=1)
    else:
        clustering = _get_MLE_clusters(gibbs_clusters, mle_fraction)

    clustering_as_partition_of_indices = [np.where(clustering==cluster_index)[0]\
                                            for cluster_index in np.unique(clustering)]
    return clustering_as_partition_of_indices

def _rw_gibbs(p, k, tau=1, n_iter=10, init=None, maximum_likelihood=True, transition_parameter=0.75):
    n = p.shape[0]
    T = np.linalg.matrix_power(p, tau)
    clusters = np.zeros((n_iter+1,n,k))
    if init is None:
        for i in range(n):
            clusters[0, i, np.random.choice(k)] = 1
    else:
        for i in range(n):
            clusters[0, i, init[i]] = 1

    # RANDOM WALK:
    for it in range(n_iter):
        # next_step[i,j] gives the probability of the i'th vertex landing on a vertex with color
        # j, given the coloring from the previous iteration.
        next_step = T.dot(clusters[it])
        next_step = normalize_rows(next_step, norm=1) # to avoid numerical issues
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
            clusters[it+1][vertex, color] = 1
        if maximum_likelihood and np.all(clusters[it+1]==clusters[it]):
            return clusters[:it+2]

    if maximum_likelihood:
        raise UserWarning("gibbs metastable clustering didn't converge, try increasing max_iter\
                            or changing split criteria")
    return clusters


def _get_MLE_clusters(clusters, fraction=0.1, n_samples=None):
    # clusters is an array of shape (n_iter, n, k), s.t. clusters[m,i,j]==1 iff i was colored j
    # in iteration m
    if n_samples is None:
        n_iter = clusters.shape[0]
        n_samples = int(np.ceil(n_iter * fraction))
    sample = np.sum(clusters[-n_samples:], axis=0) # sample[i,j] is the number of times i was
                                                   #colored j in the final n_samples iterations
    return np.argmax(sample, axis=1)
