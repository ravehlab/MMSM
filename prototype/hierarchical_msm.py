import warnings
import time
import numpy as np
from HMSM.prototype.hierarchical_msm_tree import HierarchicalMSMTree
from HMSM.prototype.hmsm_config import get_default_config

def _get_stop_condition(max_time, max_samples, max_timescale):
    # TODO: consider moving somewhere more logical
    # TODO: Add at least a brief doc
    stop_conditions = []

    if max_time is not None:
        start_time = time.time()
        stop_conditions.append(lambda n_samples, current_time, timescale: \
                                    current_time > start_time + max_time)
    if max_samples is not None:
        stop_conditions.append(lambda n_samples, current_time, timescale: \
                                    n_samples > max_samples)
    if max_timescale is not None:
        stop_conditions.append(lambda n_samples, current_time, timescale: \
                                    timescale > max_timescale)
    return lambda *args: np.any([condition(*args) for condition in stop_conditions])

class HierarchicalMSM: # TODO: change to HierarchicalMSM everywhere
    """HierarchicalMSM.
    This class is used to build and manage a hierarchical MSM over some energy landscape, as will
    be described in Clein et al. 2021. It relies on the provided sampler to explore microstates, that
    are stored at the vertices of lowest (highest-resolution) level in the MSM. It then creates
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
            The timestep resolution of the discrete trajectories used to estimate the HMSM, as 
            multiples of the dt of the sampler; i.e. if base_tau==5, and sampler.dt==2e-15, then
            the trajectories used to estimate the HMSM will be in timesteps of 5*2e-15=10e-15
            seconds.
        split_condition: function
            A function that takes an HierarchicalMSMVertex and returns True if this vertex should
            split into two or more vertices
        split_method: function
            A function that takes an HierarchicalMSMVertex v, and returns a tuple
            (partition, taus), where partition is a partition of the vertices of v, and taus is
            a list of the tau of the MSMs representing the partition parts respectively.
        sample_method: function
            A function that takes an HierarchicalMSMVertex v, and returns the id of one of its
            children in the tree.
        parent_update_threshold: double
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
    1 second (2e15*sampler.dt = 2e15*2e-15 = 1 second):

    >>> hmsm.expand(max_timescale=2e15) # TODO: disambiguate the interface - make it crystal clear what the expected result would be
    >>> tree.get_longest_timescale(dt=sampler.dt)
    1.52794


    """


    def __init__(self, sampler, start_points, **config_kwargs):
        self.sampler = sampler
        self.config = get_default_config()
        self.config.update(config_kwargs)
        self.hmsm_tree = HierarchicalMSMTree(self.config)
        self._init_sample(start_points)
        

    def _init_sample(self, start_points):
        dtrajs = self.sampler.get_initial_sample(start_points,\
                                                 self.config["n_samples"],\
                                                 self.config["sample_len"],\
                                                 self.config["base_tau"])
        self.hmsm_tree.update_model_from_trajectories(dtrajs)


    @property
    def batch_size(self):
        return self.config["n_microstates"] * self.config["n_samples"] * self.config["sample_len"]

    # TODO: update to expand() everywhere
    # TODO: (long term, not now) - add expand by confidence interval, maybe there should be an expand vs. exploit mode, or jsut
    #        according to parametersw
    def expand(self, max_cputime=None, max_samples=None, max_timescale=None):
        """
        Estimate an HMSM by sampling from the sampler.

        Parameters
        ----------
        max_cputime : # TODO: update to max_cputime everywhere
            Maximum cpu time to run in seconds.
        max_samples :
            Maximum number of samples to use.
        max_timescale :
            Minimum timescale of the full HMSM, after which to stop sampling.
        """

        if max_time is max_samples is max_timescale is None:
            warnings.warn("At least one of the parameters max_time, max_samples, or \
                              max_timescale must be given")
            return

        stop_condition = _get_stop_condition(max_time, max_samples, max_timescale)
        n_samples = 0
        timescale = np.inf
        batch_size = self.batch_size

        while not stop_condition(n_samples, time.time(), timescale):
            microstates = self.hmsm_tree.sample_microstate(n_samples=self.config["n_microstates"])
            self._batch_sample_and_estimate(microstates)
            n_samples += batch_size
            timescale = self.hmsm_tree.get_longest_timescale(self.sampler.dt)

    def _batch_sample_and_expand(self, microstates):
        dtrajs = self.sampler.sample_from_microstates(microstates,\
                                                      self.config["n_samples"],\
                                                      self.config["sample_len"],
                                                      self.config["base_tau"])
        self.hmsm_tree.update_model_from_trajectories(dtrajs)
