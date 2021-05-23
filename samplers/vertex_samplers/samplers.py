
import numpy as np
from HMSM import HMSMConfig

def get_vertex_sampler(config:HMSMConfig):
    if config.vertex_sampler in ("auto", "uncertainty_minimization"):
        return uncertainty_minimization
    elif config.vertex_sampler == "uniform":
        return uniform_sample
    else:
        raise NotImplementedError(f"Optimizer {config.vertex_sampler} not implemented.")

def uniform_sample(hmsm, n_samples):
    """Samples one of this vertices children uniformly.
    """
    uniformly_chosen_child = np.random.choice(hmsm.n, size=n_samples)
    return hmsm.children[uniformly_chosen_child]

def uncertainty_minimization(hmsm, n_samples):
    children = hmsm.children
    eps = 1e-12
    p = np.ndarray(hmsm.n)
    for i, child in enumerate(children):
        p[i] = max(np.exp(-hmsm.tree.get_n_samples(child)/2), eps)
    p = p/np.sum(p)
    p = np.nan_to_num(p)
    return np.random.choice(children, size=n_samples, p=p)
