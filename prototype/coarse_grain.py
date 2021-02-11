"""K-Centers clustering"""

# Author: Kessem Clein <kessem.clein@mail.huji.ac.il>

import numpy as np
from HMSM.util.clustering import k_centers, extend_k_centers
from HMSM.util import util

__all__ = ["KCentersCoarseGrain"]

class KCentersCoarseGrain:
    """K-Centers clustering.

    
    Parameters
    ----------
    cutoff : float
        The maximum radius of a single cluster
    representative_sample_size : int, optional
        The maximum number of representatives to keep from each cluster
    """

    def __init__(self, cutoff, representative_sample_size=10):
        self._centers = []
        self._nearest_neighbors = None
        self._k_centers_initiated = False
        self._cluster_inx_2_id = dict()

        self._representative_sample_size = representative_sample_size
        self._representatives = dict()

        self._cluster_count = util.count_dict()
        self.cutoff = cutoff

    @property
    def n_clusters(self):
        return len(self._centers)

    @property
    def centers(self):
        return self._centers.copy()


    def get_coarse_grained_clusters(self, data : np.ndarray):
        """Get the cluster ids of data points.

        Parameters
        ----------
        data : np.ndarray of shape (n_samples, n_features)
            The data points to cluster

        Returns
        -------
        labels : list of int
            A list of length n_samples, such that labels[i] is the label of the point data[i]
        """
        n_clusters = self.n_clusters
        if self._k_centers_initiated:
            self._nearest_neighbors, clusters, self._centers = extend_k_centers(data, \
                                                                self._nearest_neighbors, \
                                                                self._centers, self.cutoff)
        else:
            self._nearest_neighbors, clusters, self._centers = k_centers(data, cutoff=self.cutoff)
            self._k_centers_initiated = True

        if np.max(clusters) > n_clusters-1:
            new_clusters = np.unique(clusters)
            new_clusters = new_clusters[np.where(new_clusters > (n_clusters-1))]
            self._add_clusters(new_clusters)

        self._sample_representatives(clusters, data)

        return self._get_ids(clusters)

    def sample_from(self, cluster_id):
        """Get a sample representative from a given cluster.
        The sample returned will be one of the points previously labeled as cluster_id, such that
        the probability of each point that was seen from this cluster_id to be sampled is 1/n,
        where n is the number of points from this cluster that were observed.

        Parameters
        ----------
        cluster_id : int
            The id of the cluster to sample from

        Returns
        -------
        x : np.ndarray of shape (n_features)
            The sampled point

        Notes
        -----
        While the distribution of each individual sample is uniform over all seen samples apriori,
        the distribution of more than one sample is not uniform. This is because a finite set of
        representatives is kept from each cluster.
        The greater the parameter representative_sample_size is, the closer this distribution is
        to uniform.
        """
        random_index = np.random.randint(len(self._representatives[cluster_id]))
        return self._representatives[cluster_id][random_index]


    def _get_ids(self, cluster_indices):
        return [self._cluster_inx_2_id[inx] for inx in cluster_indices]

    def _add_clusters(self, new_clusters):
        for cluster in new_clusters:
            if self._cluster_inx_2_id.get(cluster):
                continue
            self._cluster_inx_2_id[cluster] = util.get_unique_id()

    def _sample_representatives(self, clusters, data):
        """
        Keep representative samples for each cluster, such that the probability of a representative
        x from cluster j being sampled (when sampling from the representatives) is 1/n, where n is
        the number of points x' that have been observed in j.
        """
        for i, cluster_index in enumerate(clusters):
            cluster_id = self._cluster_inx_2_id[cluster_index]
            self._cluster_count[cluster_id] += 1
            if not self._representatives.get(cluster_id):
                # If this is the first observation of this cluster
                self._representatives[cluster_id] = [data[i]]

            sample_probability = self._representative_sample_size/self._cluster_count[cluster_id]
            if np.random.random() < sample_probability:
                # with probability r/n, keep this point. It can be proven by induction
                # that this gives the desired property (where r is the number of representatives).
                if self._cluster_count[cluster_id] < self._representative_sample_size:
                    self._representatives[cluster_id].append(data[i])
                else:
                    random_index = np.random.choice(self._representative_sample_size)
                    self._representatives[cluster_id][random_index] = data[i]
