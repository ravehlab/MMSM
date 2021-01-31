from functools import reduce
import numpy as np
import matplotlib as mpl
from numpy.linalg import LinAlgError
from sklearn.cluster import MiniBatchKMeans

sample_from_ndarray = lambda array, n : array[np.random.choice(array.shape[0], n, replace=False)] \
                                        if n<=array.shape[0] else array.copy()

def _assert_2d(array):
    if array.ndim != 2:
        raise LinAlgError('%d-dimensional array given. Array must be '
                          'two-dimensional' % array.ndim) 

def _assert_finite(array):
    if not np.isfinite(array).all():
        raise LinAlgError("Array must not contain infs or NaNs")

def _assert_stacked_square(array):
    m, n = array.shape[-2:]
    if m != n:
        raise LinAlgError('Last 2 dimensions of the array must be square')

def _assert_stochastic(array, axis=1):
    if not np.all(np.isclose(np.sum(array, axis=1), 1)):
        raise LinAlgError(f"Array must sum to 1 across axis {axis}")

def _assert_valid_transition_matrix(array):
    _assert_2d(array)
    _assert_stacked_square(array)
    _assert_stochastic(array)

def stationary_distribution(transition_matrix, max_iter=1e2) -> np.ndarray:
    # make sure it's a stochastic matrix
    _assert_valid_transition_matrix(transition_matrix)

    z = transition_matrix.copy()
    for i in range(int(max_iter)):
        z = np.matmul(z,z)
        if np.all(np.isclose(z[0], z)):
            return (z[0])
    raise UserWarning("Graph is not ergodic; The distribution returned assumes uniform distribution over starting vertices.")
    return np.mean(z, axis=0)

    

def distance_matrix(X, Y=None, self_distance=np.inf):
    """distance_matrix.
    Calculate a pointwise euclidean distance matrix between X and Y. If Y is None returns
    the pairwise distances between the points in X.
    """
    if Y is None:
        dist = np.array([np.linalg.norm(X-X[j], axis=1) for j in range(X.shape[0])])
        np.fill_diagonal(dist, self_distance)
        return dist

    if not isinstance(Y, np.ndarray):
        y = np.array(Y)
    if Y.ndim==1:
        if X.shape[-1] != Y.shape[-1]:
            raise LinAlgError(f"X and Y last dimensions don't match: {X.shape}, {Y.shape}")
        dist = np.linalg.norm(X-Y, axis=1)
    else:
        if X.shape[-1] != Y.shape[-1]:
            raise LinAlgError(f"X and Y last dimensions don't match: {X.shape}, {Y.shape}")
        dist = np.array([np.linalg.norm(X-Y[j], axis=1) for j in range(Y.shape[0])])
    _assert_2d(dist)
    return dist

def min_connected_graph(distances):
    _assert_2d(distances)
    if np.any(distances<0):
        raise LinAlgError("Negative distances not supported")

    k = distances.shape[0]
    connectivity = np.eye(k)
    # first connect all the nearest neighbors
    for j in range(k):
        nearest_neighbor = np.argmin(distances[j])
        connectivity[j, nearest_neighbor] = 1
        connectivity[nearest_neighbor, j] = 1
    cc = connected_components(connectivity)
    # now connect the seperate components by their minimal edges
    while cc is not True:
        # find the shortest edge from the first cc to a vertex outside it
        first_component = np.where(cc[0])[0]
        all_the_rest = np.where(1-cc[0])[0]
        inter_component_distances = distances[np.ix_(first_component, all_the_rest)]
        shortest_distance = np.argmin(inter_component_distances)
        u,v = np.unravel_index(shortest_distance, inter_component_distances.shape) # numpy index stuff
        u = first_component[u] # get the absolute index of u 
        v = all_the_rest[v] # get the absolute index of u 
        # now connect u and v
        connectivity[u,v] = 1
        connectivity[v,u] = 1
        cc = connected_components(connectivity)
    return connectivity

def connected_components(E):
    
    #E should be a connectivity graph of an aperiodic directed graph.
    # returns true if the graph is strongly connected, otherwise returns cc an ndarray of bool with
    # shape (n,k) where n is the number of connected components, and k is the number of vertices, and 
    # cc[i,j]==True if and only if j is in the i'th component
    _assert_2d(E)
    k = E.shape[0]
    cc = np.abs(np.linalg.matrix_power(E, k))
    cc = np.unique(cc>0, axis=0)
    if np.all(cc):
        return True
    else:
        return cc

def normalize_rows(matrix:np.ndarray, norm:int=2)->np.ndarray:
    _assert_finite(matrix)
    normalized = matrix/np.linalg.norm(matrix, norm, axis=1)[:, None]
    return np.nan_to_num(normalized)


def normalized_laplacian(P):
    N = P.shape[0]
    D = np.diag(np.power(np.sum(P, axis=1), -0.5)) #D^-1/2
    L = np.eye(N) - reduce(np.matmul, [D, P, D])
    return L
