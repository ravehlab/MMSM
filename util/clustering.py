import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from .linalg import normalize_rows, normalized_laplacian

__all__ = ["k_centers", "extend_k_centers", "gibbs_metastable_clustering", "spectral"]


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

def gibbs_metastable_clustering(T, tau, max_k, init='spectral', manual_init=None, \
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
    gibbs_clusters = _rw_gibbs(T, max_k, tau, n_iter, initial_clustering, maximum_likelihood)
    if maximum_likelihood:
        # the last iteration _is_ the MLE, so we only need one sample
        clustering = _get_MLE_clusters(gibbs_clusters, n_samples=1) 
    else:
        clustering = _get_MLE_clusters(gibbs_clusters, mle_fraction)

    clustering_as_partition_of_indices = [np.where(clustering==cluster_index)[0]\
                                            for cluster_index in np.unique(clustering)]
    return clustering_as_partition_of_indices

def _rw_gibbs(p, k, tau=1, n_iter=10, init=None, maximum_likelihood=True):
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
                color = np.argmax(next_step[vertex])
            else:
                color = np.random.choice(k, p=next_step[vertex])
            clusters[it+1][vertex, color] = 1
        if maximum_likelihood and np.all(clusters[it+1]==clusters[it]):
            return clusters[:it+2]

    if maximum_likelihood:
        raise UserWarning("gibbs_metastable_clustering didn't converge, try increasing max_iter\
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
