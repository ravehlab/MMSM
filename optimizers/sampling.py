
import numpy as np
from HMSM import HMSMConfig

def get_optimizer(config:HMSMConfig):
    if config.sampling_optimizer in ("auto", "uncertainty_minimization"):
        return uncertainty_minimization
    elif config.sampling_optimizer == "uniform":
        return uniform_sample
    else:
        raise NotImplementedError(f"Optimizer {config.sampling_optimizer} not implemented.")

def uniform_sample(hmsm):
    """Samples one of this vertices children uniformly.
    """
    uniformly_chosen_child = np.random.choice(hmsm.n)
    return hmsm.children[uniformly_chosen_child]

def uncertainty_minimization(hmsm):
    children = hmsm.children
    eps = 1e-12
    p = np.ndarray(hmsm.n)
    for i, child in enumerate(children):
        p[i] = max(np.exp(-hmsm.tree.get_n_samples(child)/2), eps)
    p = p/np.sum(p)
    p = np.nan_to_num(p)
    return np.random.choice(children, p=p)
