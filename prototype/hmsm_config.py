import numpy as np
from HMSM.util import clustering

class HMSM_config:

    def __init__(self, split_condition, split_method, sample_method):

        self.split_condition = split_condition
        self.split_method = split_method
        self.sample_method = sample_method

def two_fold_gibbs_split(hmsm):
    T = hmsm.T
    tau = hmsm.tau
    max_k = int(np.sqrt(hmsm.n))
    partition = clustering.gibbs_metastable_clustering(T, 2*tau, max_k)
    taus = [tau]*len(partition)
    return partition, taus

def get_size_or_timescale_split_condition(max_size, max_timescale):
    def _split_condition(hmsm):
        return hmsm.timescale > max_timescale or hmsm.n > max_size
    return _split_condition

def uniform_sample(hmsm):
    uniformly_chosen_child = np.random.choice(hmsm.n)
    return hmsm.children[uniformly_chosen_child]
