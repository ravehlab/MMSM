from functools import reduce
import numpy as np
from sklearn.neighbors import NearestNeighbors

__all__ = ["k_centers", "extend_k_centers"]

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

def k_centers(data : np.ndarray, k=None, cutoff=None):
    """k_centers.

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
    """
    if k is None and cutoff is None:
        raise ValueError("at least one of k or cutoff must be defined")
    N = data.shape[0]
    d = data.shape[1]
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
        i += 1
        _k_centers_step(i, data, clusters, centers, distances)
    return NearestNeighbors(n_neighbors=1, algorithm='auto').fit(centers), clusters

def normalized_laplacian(P):
    N = P.shape[0]
    D = np.diag(np.power(np.sum(P, axis=1), -0.5)) #D^-1/2
    L = np.eye(N) - reduce(np.matmul, [D, P, D])
    return L

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
        e_vals, e_vecs = np.linalg.eigh(L) # gen the eigenvectors
    if return_eig:
        eig = e_vals, e_vecs
    
    k_smallest = np.argpartition(e_vals, k)[:k]
    e_vecs = e_vecs.T[k_smallest]
    e_vecs = e_vecs/np.linalg.norm(e_vecs.T, axis=1) # project them to the unit circle
    clusters = KMeans(n_clusters=k).fit_predict(e_vecs.T)
    if return_eig:
        return clusters, eig
    return clusters
