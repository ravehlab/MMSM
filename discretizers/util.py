import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from HMSM.util.linalg import normalize_rows, normalized_laplacian

__all__ = ["gibbs_metastable_clustering", "spectral"]

def spectral(P, k, eig=None, return_eig=False):
    """
    Cluster the data into k clusters using the spectral clustering algorithm.
    :param P: A transition probability matrix
    :param k: The number of desired clusters.
    :return: clustering an np.ndarray such that clustering[i] is the assignment of the i'th vertex
    """
    # Sometimes we may call this function many times on the same X, so we want to be able to
    # calculate the eigenvectors once in advance, because that may be the heaviest part computationally
    if eig != None:
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

