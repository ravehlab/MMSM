"""K-Centers clustering"""

# Author: Kessem Clein <kessem.clein@mail.huji.ac.il>

import numpy as np
from sklearn.neighbors import NearestNeighbors
from HMSM.util.util import count_dict, get_unique_id
from HMSM.discretizers import BaseDiscretizer

__all__ = ["KCentersDiscretizer"]

def k_centers(data : np.ndarray, k=None, cutoff=None):
    """k_centers.
    Implementation of k-centers clustering algorithm [1]_.


    Parameters
    ----------
    data : np.ndarray((N,d))
        a set of N datapoints with dimension d
    k :
        maximum unmber of centers
    cutoff :
        maximum distance between points and their cluster centers

    Returns
    -------
    (nn, clusters, centers)
    nn : sklearn.neighbors.NearestNeighbors
        a nearest neighbors object fitted on the cluster centers
    clusters : np.ndarray(N, dtype=np.int)
        the cluster assignments of the datapoints
    centers : list
        a list of the cluster centers

    References
    ----------
    [1] Gonzalez T (1985) Clustering to minimize the maximum intercluster distance. Theor Comput Sci 38:293
    """
    if k is None and cutoff is None:
        raise ValueError("at least one of k or cutoff must be defined")
    N = data.shape[0]
    clusters = np.zeros(N, dtype=np.int)
    centers = []
    distances = np.zeros(N)
    stop_conditions = []

    if k is not None:
        stop_conditions.append(lambda : len(centers) >= k)
    if cutoff is not None:
        stop_conditions.append(lambda : np.max(distances) <= cutoff)

    stop_condition = lambda : np.any([check_condition() for check_condition in stop_conditions])

    centers.append(data[np.random.choice(N)])
    distances = np.linalg.norm(data - centers[0], axis=1)

    i = 0
    while not stop_condition():
        i += 1
        _k_centers_step(i, data, clusters, centers, distances)
    return NearestNeighbors(n_neighbors=1, algorithm='auto').fit(centers), clusters, centers

def extend_k_centers(data, nn, centers, cutoff):
    """extend_k_centers.
    Given a set of center points, and a new set of data points, extends the list of centers
    until all the new points are within a given distance from a cluster center.

    Parameters
    ----------
    data :
        data
    nn : sklearn.neighbors.NearestNeighbors
        a nearest neighbors object fitted on the cluster centers
    centers : list
        centers
    cutoff :
        cutoff

    Returns
    -------
    (nn, clusters, centers)
    nn : sklearn.neighbors.NearestNeighbors
        a nearest neighbors object fitted on the cluster centers
    clusters : np.ndarray(N, dtype=np.int)
        the cluster assignments of the datapoints
    centers : list
        a list of the cluster centers
    """
    distances, clusters = nn.kneighbors(data)
    distances = distances.squeeze()
    clusters = clusters.squeeze()
    i = len(centers)

    while np.max(distances) >= cutoff:
        _k_centers_step(i, data, clusters, centers, distances)
        i += 1
    return NearestNeighbors(n_neighbors=1, algorithm='auto').fit(centers), clusters, centers

def _k_centers_step(i, data, clusters, centers, distances):
    """_k_centers_step.

    Parameters
    ----------
    i :
        the index of the next center
    data :
        the datapoints
    clusters :
        the current clustering of the data
    centers :
        the current cluster centers
    distances :
        the current distances between the datapoints and their current cluster centers
    """
    new_center = data[np.argmax(distances)]
    centers.append(new_center)
    dist_to_new_center = np.linalg.norm(data - new_center, axis=1)
    updated_points = np.where(dist_to_new_center < distances)[0]
    clusters[updated_points] = i
    distances[updated_points] = dist_to_new_center[updated_points]

class KCentersDiscretizer(BaseDiscretizer):
    """K-Centers clustering.


    Parameters
    ----------
    cutoff : float
        The maximum radius of a single cluster
    representative_sample_size : int, optional
        The maximum number of representatives to keep from each cluster

    Examples
    --------
    TODO
    """

    def __init__(self, cutoff, representative_sample_size=10):
        self._centers = []
        self._nearest_neighbors = None
        self._k_centers_initiated = False
        self._cluster_inx_2_id = dict()
        self._id_2_cluster_inx = dict()

        self._representative_sample_size = representative_sample_size
        self._representatives = dict()

        self._cluster_count = count_dict()
        self.cutoff = cutoff

    @property
    def n_states(self):
        return len(self._centers)

    @property
    def centers(self):
        return np.array(self._centers)

    def get_centers_by_ids(self, ids):
        indices = np.array([self._id_2_cluster_inx[id] for id in ids], dtype=int)
        return self.centers[indices]

    def get_coarse_grained_states(self, data : np.ndarray):
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
        n_clusters = self.n_states
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
            uid = get_unique_id()
            self._cluster_inx_2_id[cluster] = uid
            self._id_2_cluster_inx[uid] = cluster

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
