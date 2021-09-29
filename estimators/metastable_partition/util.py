
import numpy as np
from sklearn.cluster import KMeans
from HMSM.util.linalg import normalize_rows, normalized_laplacian

def spectral(P, k, eig=None, return_eig=False):
    """
    Cluster the data into k clusters using the spectral clustering algorithm.
    :param P: A transition probability matrix
    :param k: The number of desired clusters.
    :return: clustering an np.ndarray such that clustering[i] is the assignment of the i'th vertex
    """
    # Sometimes we may call this function many times on the same X, so we want to be able to
    # calculate the eigenvectors once in advance, because that may be the heaviest part computationally
    if eig is not None:
        e_vals, e_vecs = eig
    else:
        L = normalized_laplacian(P)
        e_vals, e_vecs = np.linalg.eig(L) # gen the eigenvectors
    if return_eig:
        eig = e_vals, e_vecs

    e_vals = np.real(e_vals)
    e_vecs = np.real(e_vecs)
    k_smallest = np.argpartition(e_vals, min(k, len(e_vals)-1))[:k]
    projection = e_vecs[:,k_smallest]
    projection = normalize_rows(projection, norm=2)
    clusters = KMeans(n_clusters=k).fit_predict(projection)
    if return_eig:
        return clusters, eig
    return clusters

def get_size_or_timescale_split_condition(max_size=2048, max_timescale=32):
    """Get a function which checks if a vertices size or timescale are larger than some constant,
    provided as arguments to this function.
    """
    return size_or_timescale_split_condition(max_timescale, max_size)

class size_or_timescale_split_condition:
    def __init__(self, max_timescale, max_size):
        self.max_timescale = max_timescale
        self.max_size = max_size

    def __call__(self, vertex):
        return vertex.timescale/vertex.tau > self.max_timescale or vertex.n > self.max_size

