import functools
import numpy as np
from HMSM.discretizers import BaseDiscretizer


def elpair(x : int,y : int) -> int:
    """Szudzik's Elegant Pairing function from http://szudzik.com/ElegantPairing.pdf
    An bijective map NxN->N, elunpair is the inverse function.
    """
    if x>=y:
        return x*x + x + y
    else:
        return y*y + x

def elunpair(z : int) -> (int, int):
    """The inverse of elpair"""
    sqrt_z = np.floor(np.sqrt(z))
    sqrtdiff_z = (z-sqrt_z*sqrt_z)
    if sqrtdiff_z<sqrt_z:
        return sqrtdiff_z, sqrt_z
    else:
        return sqrt_z, sqrtdiff_z - sqrt_z

@functools.lru_cache(maxsize=256)
def signed_pair(x : int,y : int) -> int:
    """
    A bijective map ZxZ->Z
    The sign of signed_pair(x, y) is the sign of x, and signed_pair(x,y)%2==1 if y<0. 
    Finally, signed_pair(x,y)//2==elpair(|x|,|y|).
    """
    x_sign = 1 if x >= 0 else -1
    y_sign = 0 if y >= 0 else 1
    return elpair(abs(x),abs(y))*2*x_sign + y_sign

@functools.lru_cache(maxsize=256)
def signed_unpair(z : int) -> (int, int):
    """The inverse of signed_pair"""
    y_sign = 1 if z%2==0 else -1
    x_sign = 1 if z >= 0 else -1
    z_ = abs(z//2)
    x, y = elunpair(z_)
    return x_sign*x, y_sign*y

# vectorize the pair functions so they can be applied to numpy arrays:
grid2id = np.vectorize(lambda xy : signed_pair(*xy), signature="(2)->()", otypes=[np.int64])
id2grid = np.vectorize(lambda z : np.array(signed_unpair(z)), signature="()->(2)", otypes=[np.int64])

class TwoDimensionalGridDiscretizer(BaseDiscretizer):
    """GridDiscretizer.
    Provides coarse graining of configuration space into a grid.

    Parameters
    ----------
    grid_size : float
        The side length of each microstate
    """

    def __init__(self, grid_size : float, representative_sample_size=10):
        super().__init__(representative_sample_size)
        self.__grid_size = grid_size
        self.active_states = set()

    @property
    def n_states(self):
        return len(self.active_states)

    @property
    def grid_size(self):
        #this must be immutable
        return self.__grid_size

    def _coarse_grain_states(self, data : np.ndarray):
        """_coarse_grain_states.

        Parameters
        ----------
        data : np.ndarray((N,d))
            a set of N datapoints with dimension d

        Returns
        -------
        labels : list of int
            A list of length n_samples, such that labels[i] is the label of the point data[i]
        """
        N = data.shape[0]
        grids = data//self.grid_size
        cluster_ids = grid2id(grids)
        self.active_states = self.active_states.union(cluster_ids)
        return list(cluster_ids)

    def get_centers_by_ids(self, cluster_ids):
        grid_corners = id2grid(np.fromiter(cluster_ids, dtype=int))
        return grid_corners*self.grid_size + (self.grid_size/2) #grid corner -> grid center
