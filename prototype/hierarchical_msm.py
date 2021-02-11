import warnings
import time
import numpy as np
from HMSM.prototype.hierarchical_msm_tree import HierarchicalMSMTree
from HMSM.prototype.hmsm_config import get_default_config
from HMSM.util import util

class HierarchicalMSM:
    """HierarchicalMSM.
    This class is used to build and manage a hierarchical MSM over some energy landscape, as will
    be described in Clein et al. 2021. It relies on the provided sampler to explore microstates, 
    that are stored at the vertices of lowest (highest-resolution) level in the MSM. It then creates
    a hierarchical (multiscale) map of the states in the configuration space, and stores it in a 
    HierarchicalMSMTree data structure.

    Parameters
    ----------
    sampler: DiscreteSampler
        A wrapper for some process that samples a sequences of microstates from a Markov-chain
        over a discrete set of states and a discrete time step. 
    start_points: Iterable
        A set of points from which to initiate the HierarchicalMSM at the highest-resolution level.

    Other Parameters
    ----------------
    **config_kwargs: optional
        Any parameters for the model. Here is a list of parameters used by the model:

        n_microstates: int
            How many microstates to sample from in a single batch of samples
        n_samples: int
            How many samples to take from each microstate in a single batch of samples
        sample_len: int
            The length of each sample in a single batch of samples
        base_tau: int
            The number of sampler timesteps to skip between each sample used to estimate the HMSM.
            For example if base_tau is 5, and the sampler creates a trajectory (x0, ..., x15), then
            the sub-sampled trajectory (x0, x5, x10, x15) will be used to estimate transition rates
            of the HMSM.
        split_condition: callable
            A function that takes an HierarchicalMSMVertex and returns True if this vertex should
            split into two or more vertices
        split_method: callable
            A function that takes an HierarchicalMSMVertex v, and returns a tuple
            (partition, taus), where partition is a partition of the vertices of v, and taus is
            a list of the tau of the MSMs representing the partition parts respectively.
        sample_method: callable
            A function that takes an HierarchicalMSMVertex, and returns the id of one of its
            children in the tree.
        parent_update_threshold: float
            The minimum fraction of a change in a transition probability between vertices that will
            trigger the parent of the vertices to update its transition matrix.


    Examples
    --------
    # TODO: declare (possibly in README) what's the standard default units being used everywhere,
    #       e.g. length - angstroms, dt - seconds, force-function - kcal_per_mol_per_angstroms
    >>> sampler = BrownianDynamicsSampler(force_function, dim=2, dt=2e-15, kT=1)
    >>> hmsm = HierarchicalMSM(sampler, split_method=two_fold_gibbs_split)

    sample and update in batches for 1 minute in CPU time:

    >>> tree = hmsm.expand(max_cputime=60)

    Now we can do some analysis, visualizations, etc, on the tree:
    
    TODO: provide example

    Continue sampling until the timescale of the longest process described by the HMSM is at least
    1 second:

    >>> hmsm.expand(min_timescale_sec=1)         # tree points to the HierarchicalMSMs tree data
    >>> timescale = tree.get_longest_timescale() # structure, so it was updated by hmsm.expand 
    >>> timescale_in_seconds = timescale * hmsm.timestep_in_seconds
    >>> print(f"The timescale of the slowest process described by this HMSM is {timescale_in_seconds:.1f} seconds")
    The timescale of the slowest process described by this HMSM is 1.2 seconds
    """


    def __init__(self, sampler, start_points, **config_kwargs):
        self.sampler = sampler
        self.config = get_default_config()
        self.config.update(config_kwargs)
        self.hmsm_tree = HierarchicalMSMTree(self.config)
        self.n_samples = 0
        self._effective_timestep_seconds = self.sampler.dt * self.config["base_tau"]
        self._init_sample(start_points)
        

    def _init_sample(self, start_points):
        dtrajs = self.sampler.get_initial_sample(start_points,\
                                                 self.config["n_samples"],\
                                                 self.config["sample_len"],\
                                                 self.config["base_tau"])
        self.hmsm_tree.update_model_from_trajectories(dtrajs)
        self.n_samples += self.batch_size


    @property
    def batch_size(self):
        return self.config["n_microstates"] * self.config["n_samples"] * self.config["sample_len"]

    @property
    def timestep_in_seconds(self):
        return self._effective_timestep_seconds

    # TODO: (long term, not now) - add expand by confidence interval, maybe there should be an expand vs. exploit mode, or jsut
    #        according to parametersw
    def expand(self, max_cputime=np.inf, max_samples=np.inf, min_timescale_sec=np.inf):
        """
        Estimate an HMSM by sampling from the sampler.

        Parameters
        ----------
        max_cputime : int or float
            Maximum cpu time to run in seconds.
        max_samples : int 
            Maximum number of samples to use.
        min_timescale_sec :
            Minimum timescale of the full HMSM, after which to stop sampling.
        """

        if max_cputime == max_samples == min_timescale_sec == np.inf:
            warnings.warn("At least one of the parameters max_cputime, max_samples, or \
                              min_timescale_sec must be given")
            return

        max_values = {'n_samples' : max_samples, 'timescale' : min_timescale_sec}
        stop_condition = util.get_threshold_check_function(max_values, max_time=max_cputime)
        n_samples = 0
        timescale = np.inf
        batch_size = self.batch_size

        while not stop_condition(n_samples=n_samples, timescale=timescale):
            microstates = self.hmsm_tree.sample_microstate(n_samples=self.config["n_microstates"])
            self._batch_sample_and_expand(microstates)
            n_samples += batch_size
            self.n_samples += batch_size
            timescale = self.hmsm_tree.get_longest_timescale() * self.timestep_in_seconds
            print(f"Samples used: {self.n_samples}, timescale: {timescale}")

    def _batch_sample_and_expand(self, microstates):
        dtrajs = self.sampler.sample_from_microstates(microstates,\
                                                      self.config["n_samples"],\
                                                      self.config["sample_len"],
                                                      self.config["base_tau"])
        self.hmsm_tree.update_model_from_trajectories(dtrajs)
