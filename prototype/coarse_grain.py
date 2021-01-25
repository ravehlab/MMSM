from HMSM.util.clustering import k_centers, extend_k_centers

class K_CentersCoarseGrain:

    def __init__(self):
        self.nearest_neighbors = None
        self.centers = None
        self.representatives = dict()
