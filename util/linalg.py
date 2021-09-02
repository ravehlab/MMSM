from functools import reduce
import warnings
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
    elif array.size == 0:
        raise LinAlgError('Input array has size 0')

def _assert_finite(array):
    if not np.isfinite(array).all():
        raise LinAlgError("Array must not contain infs or NaNs")

def _assert_stacked_square(array):
    m, n = array.shape[-2:]
    if m != n:
        raise LinAlgError('Last 2 dimensions of the array must be square')

def _assert_stochastic(array, axis=-1):
    if np.any(array < 0):
        raise LinAlgError(f"Array can not have negative values")
    if not np.all(np.isclose(np.sum(array, axis=axis), 1)):
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
    warnings.warn("Graph is not ergodic; The distribution returned assumes uniform \
                   distribution over starting vertices.")
    return np.mean(z, axis=0)

def normalize_rows(matrix:np.ndarray, norm:int=2)->np.ndarray:
    """
    Normalize a matrix across rows (axis 1).

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to normalize
    norm : int
        Which norm to normalize by. Default is 2 (Euclidean norm).

    Returns
    -------
    normalized_matrix : np.ndarray
        A copy of the matrix, normalized across the rows.

    """
    _assert_finite(matrix)
    normalized = matrix/np.linalg.norm(matrix, norm, axis=1)[:, None]
    return np.nan_to_num(normalized, posinf=1.)


def normalized_laplacian(P):
    N = P.shape[0]
    D = np.diag(np.power(np.sum(P, axis=1), -0.5)) #D^-1/2
    L = np.eye(N) - reduce(np.matmul, [D, P, D])
    return L


def get_longest_timescale(P, tau):
    """Get the implied timescale of the second largest eigenvector of a transition matrix P with 
    time resolution tau
    """
    _assert_valid_transition_matrix(P)
    w,v = np.linalg.eig(P)
    if len(w) == 2:
        lambda_2 = min(w)
    else:
        two_largest = np.argpartition(-w, 2)[:2]
        lambda_2 = w[two_largest][1]
        if np.isclose(lambda_2, 1):
            return np.inf

    t_2 = np.abs(tau/np.log(lambda_2))
    return t_2

