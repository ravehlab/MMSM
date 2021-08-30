from functools import lru_cache
from collections import defaultdict
from scipy.stats import multivariate_normal
from tqdm import tqdm
import numpy as np
from HMSM.discretizers import TwoDimensionalGridDiscretizer

__all__ = ["get_transition_matrix"]

def make_2d(T, limit, grid_size, grid_discretizer):
    n = T.shape[0]
    T2 = np.ndarray((n*n, n*n))
    #index to grid index - translates 1D indices of T2 to 2D indices of T (which is 3D)
    i2g = lambda i: (i//n, i%n) 
    for i, j in np.ndindex(T2.shape):
        T2[i,j] = T[i2g(i)][i2g(j)]

    grid_corners = np.ndarray((n*n,2))
    for i in range(n*n):
        grid_corners[i] = index2coords(*i2g(i), limit, grid_size)

    ids = grid_discretizer.get_coarse_grained_states(grid_corners)
    return T2, ids




def get_transition_matrix(force, dt, kT, grid_discretizer, limit, subgrid_size):
    grid_size = grid_discretizer.grid_size
    force_np = np.vectorize(force, signature="(2)->(2)")
    brownian_noise = 2*np.sqrt(kT*dt)
    n = int((2*limit)//grid_size)
    # T is the transition matrix, indexed by grid coordinates.
    # i.e. T[x1,y1,x2,y2] is the probability of transitioning from square (x1,y1) to square (x2,y2)
    T = np.ndarray((n,n,n,n))
    for i,j in tqdm(np.ndindex((n,n)), total=n*n):
        x, y = index2coords(i, j, limit, grid_size)
        probs_dict = get_transition_probabilities(x,y, force_np, dt, brownian_noise,
                                                  grid_size, subgrid_size, limit)
        for (i2, j2), p in probs_dict.items():
            T[i,j,i2,j2] = p
    assert not np.any(np.isnan(T))
    return make_2d(T, limit, grid_size, grid_discretizer)


def index2coords(i, j, limit, grid_size):
    x = -limit + i*grid_size
    y = -limit + j*grid_size
    return x, y

def coords2indices(x, y, limit, grid_size):
    i = int((x+limit)/grid_size)
    j = int((y+limit)/grid_size)
    return i, j

@lru_cache()
def get_subgrid(grid_size, subgrid_size):
    return np.linspace(0, grid_size, subgrid_size+2)[1:-1]

def get_transition_probabilities(x,y, force, dt, brownian_noise, grid_size, subgrid_size, limit):
    xs = x + get_subgrid(grid_size, subgrid_size)
    ys = y + get_subgrid(grid_size, subgrid_size)
    xy = np.stack(np.meshgrid(xs, ys), axis=-1)

    p = defaultdict(float)
    after_force = (xy + dt*force(xy)) # has shape (subgrid_size,subgrid_size,2)
    after_force = after_force.reshape((-1,2)) # reshape to (subgrid_size**2, 2)
    squares, indices = get_neighboring_squares(after_force, brownian_noise, limit, grid_size)
    for loc in after_force:
        for i, neighbor in enumerate(squares):
            p[indices[i]] += get_browninan_probability(loc, neighbor, grid_size, brownian_noise)

    # normalize to yield a legal distribution:
    normalization_factor = np.sum([p[index] for index in indices])
    # edge case where the probability all goes out of the limits of the grid:
    if normalization_factor == 0: 
        p[coords2indices(x,y, limit, grid_size)] = 1.
        return p
    for index in indices:
        p[index] /= normalization_factor
        assert p[index] is not np.nan
    return p

def get_browninan_probability(loc, neighbor, grid_size, brownian_noise):
    """get_browninan_probability.

    Parameters
    ----------
    loc : np.array of shape (2,)
        Mean of the normal distribution
    neighbor : np.array of shape (2,)
        bottom left coordinates of neighboring square
    grid_size :
        grid_size
    brownian_noise :
        brownian_noise

    Returns
    -------
    """
    bottom_l = neighbor
    bottom_r = [neighbor[0] + grid_size, neighbor[1]]
    top_l = [neighbor[0], neighbor[1] + grid_size]
    top_r = neighbor + grid_size
    less_than_top_right = multivariate_normal.cdf(top_r, mean=loc, cov=brownian_noise)
    less_than_top_left = multivariate_normal.cdf(top_l, mean=loc, cov=brownian_noise)
    less_than_bot_left = multivariate_normal.cdf(bottom_l, mean=loc, cov=brownian_noise)
    less_than_bot_right = multivariate_normal.cdf(bottom_r, mean=loc, cov=brownian_noise)

    less_than_left_between_top_and_bottom = less_than_top_left - less_than_bot_left
    return less_than_top_right - less_than_bot_right - less_than_left_between_top_and_bottom

def get_neighboring_squares(means, brownian_noise, limit, grid_size):
    margin = 6*brownian_noise # this covers transitions with probability less than 1/5e9
    max_x = min(np.max(means[:,0]) + margin, limit - grid_size)
    min_x = max(np.min(means[:,0]) - margin, -limit)
    max_y = min(np.max(means[:,1]) + margin, limit - grid_size)
    min_y = max(np.min(means[:,1]) - margin, -limit)

    min_i, min_j = coords2indices(min_x, min_y, limit, grid_size)
    max_i, max_j = coords2indices(max_x, max_y, limit, grid_size)
    indices = [(i,j) for j in range(min_j, max_j+1) for i in range(min_i, max_i+1)]
    return np.array([index2coords(i,j, limit, grid_size) for i,j in indices]), indices
