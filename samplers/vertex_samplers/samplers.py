from msmtools.analysis import committor
from msmtools.flux import flux_production, flux_matrix
import numpy as np
from HMSM import HMSMConfig
from HMSM.util import linalg

class WeightedVertexSampler():
    def __init__(self, heuristics, weights):
        self._weights = weights
        self._heuristics = heuristics
        linalg._assert_stochastic(self._weights)


    def __call__(self, vertex, n_samples):
        return np.random.choice(vertex.children,
                                size=n_samples,
                                p=self._get_distribution(vertex) )

    def _get_distribution(self, vertex):
        distributions = np.array([heuristic(vertex) for heuristic in self._heuristics])
        if np.any(np.isnan(distributions)):
            import pdb; pdb.set_trace()
            distributions = np.array([heuristic(vertex) for heuristic in self._heuristics])
        return self._weights.dot(distributions)

def _get_sampler_by_name(name):
    if name in ("auto", "uncertainty_minimization"):
        return uncertainty_minimization
    elif name == "uniform":
        return uniform_sample
    elif name == "flux":
        return flux
    elif name == "equilibrium":
        return equilibrium
    else:
        raise NotImplementedError(f"Optimizer {name} not implemented.")


def get_vertex_sampler(config:HMSMConfig):
    if config.vertex_sampler in ("auto", "weighted"):
         heuristics = [_get_sampler_by_name(h) for h in config.sampling_heuristics]
         weights = np.array(config.sampling_heuristic_weights)
         return WeightedVertexSampler(heuristics, weights)
    else:
        raise NotImplementedError(f"Optimizer {config.vertex_sampler} not implemented.")

def equilibrium(vertex):
    return vertex.local_stationary

def uniform_sample(vertex):
    """Samples one of this vertices children uniformly.
    """
    return np.ones(vertex.n)/vertex.n

def uncertainty_minimization(vertex):
    children = vertex.children
    eps = 1e-6
    p = np.ndarray(vertex.n)
    for i, child in enumerate(children):
        p[i] = max(np.exp(-vertex.tree.get_n_samples(child)/4), eps)
    p = p/np.sum(p)
    return np.nan_to_num(p)

def flux(vertex):
    if vertex.n==1:
        return np.ones(1)
    # get source_sink_T
    n = vertex.n
    n_full = vertex.T.shape[0]
    if n_full == n:
        return vertex.local_stationary
    T = np.zeros((n_full+1, n_full+1))
    T[:n_full, :n_full] = vertex._T
    # get the local stationary distribution
    mu = np.zeros(n_full+1)
    mu[:n] = vertex.local_stationary
    # T[-1] is the source state, from there we go directoy to the local stationary distribution
    T[-1] = mu
    # T[i, -1] represents the probability that we've "mixed", i.e. gone back to the stationary
    # distribution, represented by the state with index -1, after bing in state i
    for i in range(n):
        T[i, -1] = T[i].dot(mu)
    T = linalg.normalize_rows(T, norm=1)
    source = [n_full]
    sink = np.arange(n, n_full)
    # get commitors
    qplus = committor(T, source, sink)
    qminus = 1-qplus
    # get flux network
    F = flux_matrix(T, mu, qminus, qplus)
    flux = np.abs(flux_production(F)[:n]) #TODO maybe only use positive flux?
    return flux/np.sum(flux)
