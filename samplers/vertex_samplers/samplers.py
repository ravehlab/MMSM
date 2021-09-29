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
        return self._weights.dot(distributions)

    def _get_full_distribution(self, vertex):
        local_distribution = self._get_distribution(vertex)
        # base case:
        if vertex.height == 1:
            return dict(zip(vertex.children, local_distribution))

        # recursive construction:
        global_distribution = {}
        for i, child_id in enumerate(vertex.children):
            child = vertex.tree.vertices[child_id]
            child_distribution = self._get_full_distribution(child)
            for microstate_id, local_probability in child_distribution.items():
                global_distribution[microstate_id] = (local_probability*local_distribution[i])
        return global_distribution


def _get_sampler_by_name(name):
    #TODO just use a dictionary already!
    if name in ("auto", "exploration"):
        return exploration
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

def exploration(vertex):
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
    mu_norm = mu.dot(mu)
    for i in range(n):
        T[i, -1] = 1-(mu.dot(T[i])/mu_norm)
    T = linalg.normalize_rows(T, norm=1)
    source = [n_full]
    sink = np.arange(n, n_full)
    # get commitors
    qplus = committor(T, source, sink)
    qminus = 1-qplus
    # get flux network
    F = flux_matrix(T, mu, qminus, qplus, netflux=False)
    influxes = np.array(np.sum(F[:n,:n], axis=0)).flatten()  # all that flows in, not including from the source state
    outfluxes = np.array(np.sum(F, axis=1)).flatten()[:n]  # all that flows out from the children, including to the neighboring states
    flux_vals = outfluxes - influxes
    flux_vals[flux_vals<0] = 1e-12 # We're only intrested in positive flux producers
    for i, child in enumerate(vertex.children):
        flux_vals[i] /= max(1, vertex.tree.get_n_samples(child))
    return flux_vals/np.sum(flux_vals)
    

