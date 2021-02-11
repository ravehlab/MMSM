import numpy as np
from HMSM.util import clustering

def get_default_config(**kwargs):
    config = {
              "n_microstates" : 20,
              "n_samples" : 5,
              "sample_len" : 5,
              "base_tau" : 2,
              "split_condition" : get_size_or_timescale_split_condition(),
              "split_method" : two_fold_gibbs_split,
              "sample_method" : uniform_sample,
              "parent_update_threshold" : 0.1,
              "alpha" : 1
             }
    config.update(kwargs)
    return config

# TODO: move all methods below somewhere else, probably into a misc.py or util.py file
def two_fold_gibbs_split(hmsm):
    """Get a partition of a vertex, assuming the parent's time resolution tau is twice that of
    this vertex.
    """
    n = hmsm.n
    T = hmsm.T[:n, :n]
    tau = hmsm.tau
    max_k = int(np.sqrt(hmsm.n))
    partition = clustering.gibbs_metastable_clustering(T, 2*tau, max_k)
    taus = [tau]*len(partition)
    return partition, taus

def get_size_or_timescale_split_condition(max_size=2045, max_timescale=64):
    """Get a function which checks if a vertices size or timescale are larger than some constant,
    provided as arguments to this function.
    """
    def _split_condition(hmsm):
        return hmsm.timescale/hmsm.tau > max_timescale or hmsm.n > max_size
    return _split_condition

def uniform_sample(hmsm):
    """Samples one of this vertices children uniformly.
    """
    uniformly_chosen_child = np.random.choice(hmsm.n)
    return hmsm.children[uniformly_chosen_child]
