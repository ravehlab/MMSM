from HMSM.util.clustering import k_centers, extend_k_centers
from HMSM.util import util
import numpy as np

__all__ = ["K_CentersCoarseGrain"]

class K_CentersCoarseGrain:

    def __init__(self, cutoff, representative_sample_size=10):
        self.nearest_neighbors = None
        self.centers = []
        self.nn = None
        self._k_centers_initiated = False
        self.cluster_inx_2_id = dict()

        self.representative_sample_size = representative_sample_size
        self.representatives = dict()

        self.cluster_count = util.count_dict()
        self.cutoff = cutoff

    @property
    def n_clusters(self):
        return len(self.centers)


    def get_coarse_grained_clusters(self, data : np.ndarray):
        n_clusters = self.n_clusters
        if self._k_centers_initiated:
            self.nn, clusters, self.centers = extend_k_centers(data, self.nn, \
                                                               self.centers, self.cutoff)
        else:
            self.nn, clusters, self.centers = k_centers(data, cutoff=self.cutoff)
            self._k_centers_initiated = True

        if np.max(clusters) > n_clusters-1:
            new_clusters = np.unique(clusters)
            #new_clusters = new_clusters[np.where(new_clusters > (n_clusters-1))]
            self._add_clusters(new_clusters)

        self._sample_representatives(clusters, data)

        return self._get_ids(clusters)

    def sample_from(self, microstate_id):
        random_index = np.random.randint(len(self.representatives[microstate_id]))
        return self.representatives[microstate_id][random_index]


    def _get_ids(self, cluster_indices):
        return [self.cluster_inx_2_id[inx] for inx in cluster_indices]

    def _add_clusters(self, new_clusters):
        for cluster in new_clusters:
            if self.cluster_inx_2_id.get(cluster):
                continue
            self.cluster_inx_2_id[cluster] = util.get_unique_id()

    def _sample_representatives(self, clusters, data):
        """
        Keep representative samples for each cluster, such that the probability of a representative
        x from cluster j being sampled (when sampling from the representatives) is 1/n, where n is
        the number of points x' that have been observed in j.
        """
        for i, cluster_index in enumerate(clusters):
            cluster_id = self.cluster_inx_2_id[cluster_index]
            self.cluster_count[cluster_id] += 1
            if not self.representatives.get(cluster_id):
                # If this is the first observation of this cluster
                self.representatives[cluster_id] = [data[i]]
            elif len(self.representatives[cluster_id] ) < self.representative_sample_size:
                self.representatives[cluster_id].append(data[i])
            elif np.random.random() <= (1/self.cluster_count[cluster_id]):
                # with probability 1/n, keep this point. It can be proven easily by induction
                # that this gives the desired property.
                random_index = np.random.randint(self.representative_sample_size)
                self.representatives[cluster_id][random_index] = data[i]
