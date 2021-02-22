import numpy as np
from scipy import integrate

def integrate_2d(force, xlim, ylim, resolution):
    """integrate_2d.
    Integrate a 2D function

    Parameters
    ----------
    force : callable
        A force function which takes an ndarray of shape (2,), and returns an ndarray of the same 
        shape, indicating forces in each dimension.
    xlim : tuple
        A tuple (xmin, xmax), the range on the x axis over which to integrate
    ylim : tuple
        A tuple (ymin, ymax), the range on the y axis over which to integrate
    resolution : float
        The resolution of the output result

    Returns
    -------
    X, Y : ndarrays
        The meshgrid of coordinates of the resulting integration
    I : ndarray
        The result of the integration. For indices (i,j), I[i,j] is the integral of force at the
        point (X[i,j], Y[i,j])

    Examples
    --------
    Define a force function, and integrate it over the range (-10,10), with resolution 0.1
    >>> force = lambda x : np.array([x[0]**2 + x[0]*x[1], [x[1]**2 - x[0]*x[1]]])
    >>> xlim = ylim = (-10, 10)
    >>> resolution = 0.1
    >>> X, Y, I = integrate_2d(force, xlim, ylim, resolution)

    Now we can simply plot the resulting 2 dimensional integral:
    >>> plt.contour(X, Y, I)
    """
    x = np.arange(xlim[0], xlim[1], resolution)
    y = np.arange(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x,y)
    coords = np.stack([X,Y], axis=2)
    Fx = np.ndarray(X.shape)
    Fy = np.ndarray(Y.shape)
    for i,j in np.ndindex(X.shape):
        dx, dy = force(coords[i, j])
        Fx[i, j] = dx
        Fy[i, j] = dy
    
    Ix= integrate.cumtrapz(Fx, x, axis=1, initial= 0)
    Iy= integrate.cumtrapz(Fy, y, axis=0, initial= 0)
    Ix_by_first_row= np.tile(Ix[0,:], (Ix.shape[0],1))
    I= (Ix_by_first_row + Iy)
    return X, Y, I

def plot_distribution(cg, p, ax=None, **kwargs):
    x = cg.get_centers_by_ids(p.keys())
    c = np.log(list(p.values()))
    if ax is None:
        ax = plt.gca()
    return ax.scatter(*x.T, c=c, **kwargs)
