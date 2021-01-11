import numpy as np
from numpy.linalg import LinAlgError

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

def normalize_rows(matrix:np.ndarray, norm:int=2)->np.ndarray:
    _assert_finite(matrix)
    return matrix/np.linalg.norm(matrix, norm, axis=1)[:, None]
