from HMSM.util import clustering
class HMSM_config:

    def __init__(self, split_condition, split_method, sample_method):

        self.split_condition = split_condition
        self.split_method = split_method
        self.sample_method = sample_method

def gibbs_split(hmsm):
    T = hmsm.T
    tau = hmsm.tau
    max_k = int(np.sqrt(hmsm.n))
    partition = clustering.gibbs_metastable_clustering(T, tau, max_k)
    taus = #Something like: [get_tau for partition in partitions]
    
