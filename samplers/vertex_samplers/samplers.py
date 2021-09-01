from msmtools.analysis import committor
from msmtools.flux import flux_production, flux_matrix
import numpy as np
from HMSM import HMSMConfig
from HMSM.util.linalg import normalize_rows

def get_vertex_sampler(config:HMSMConfig):
    if config.vertex_sampler in ("auto", "uncertainty_minimization"):
        return uncertainty_minimization
    elif config.vertex_sampler == "uniform":
        return uniform_sample
    elif config.vertex_sampler == "flux":
        return weighted_flux_sampler
    else:
        raise NotImplementedError(f"Optimizer {config.vertex_sampler} not implemented.")

def uniform_sample(vertex, n_samples):
    """Samples one of this vertices children uniformly.
    """
    return np.random.choice(vertex.children, size=n_samples)

def uncertainty_minimization(vertex, n_samples):
    children = vertex.children
    eps = 1e-12
    p = np.ndarray(vertex.n)
    for i, child in enumerate(children):
        p[i] = max(np.exp(-vertex.tree.get_n_samples(child)/2), eps)
    p = p/np.sum(p)
    p = np.nan_to_num(p)
    return np.random.choice(children, size=n_samples, p=p)


def flux_sampler(vertex, n_samples):
    if vertex.n==1:
        return np.array(vertex.children*n_samples)
    # get source_sink_T
    n = vertex.n
    n_full = vertex.T.shape[0]
    T = np.zeros((n_full+1, n_full+1))
    T[:n_full, :n_full] = vertex._T
    # get the local stationary distribution
    mu = np.zeros(n_full+1)
    mu[:n] = vertex.get_local_stationary_distribution()
    # T[-1] is the source state, from there we go directoy to the local stationary distribution
    T[-1] = mu
    # T[i, -1] represents the probability that we've "mixed", i.e. gone back to the stationary
    # distribution, represented by the state with index -1, after bing in state i
    for i in range(n):
        T[i, -1] = T[i].dot(mu)
    T = normalize_rows(T, norm=1)
    source = [n_full]
    sink = np.arange(n, n_full)
    # get commitors
    qplus = committor(T, source, sink)
    qminus = 1-qplus
    # get flux network
    F = flux_matrix(T, mu, qminus, qplus)
    flux = np.abs(flux_production(F)[:n])
    p = flux/np.sum(flux)
    return np.random.choice(vertex.children, size=n_samples, p=p)

def weighted_flux_sampler(vertex, n_samples):
    children = vertex.children
    eps = 1e-12
    p = np.ndarray(vertex.n)
    for i, child in enumerate(children):
        p[i] = max(np.exp(-vertex.tree.get_n_samples(child)/2), eps)
    p = p/np.sum(p)
    p1 = np.nan_to_num(p)

    if vertex.n==1:
        return np.array(vertex.children*n_samples)
    # get source_sink_T
    n = vertex.n
    n_full = vertex.T.shape[0]
    T = np.zeros((n_full+1, n_full+1))
    T[:n_full, :n_full] = vertex._T
    # get the local stationary distribution
    mu = np.zeros(n_full+1)
    mu[:n] = vertex.get_local_stationary_distribution()
    # T[-1] is the source state, from there we go directoy to the local stationary distribution
    T[-1] = mu
    # T[i, -1] represents the probability that we've "mixed", i.e. gone back to the stationary
    # distribution, represented by the state with index -1, after bing in state i
    for i in range(n):
        T[i, -1] = T[i].dot(mu)
    T = normalize_rows(T, norm=1)
    source = [n_full]
    sink = np.arange(n, n_full)
    # get commitors
    qplus = committor(T, source, sink)
    qminus = 1-qplus
    # get flux network
    F = flux_matrix(T, mu, qminus, qplus)
    flux = np.abs(flux_production(F)[:n])
    p2 = flux/np.sum(flux)
    
    alpha = 0.9
    p = p1*alpha + p2*(1-alpha)
    p /=np.sum(p)
    return np.random.choice(vertex.children, size=n_samples, p=p)

