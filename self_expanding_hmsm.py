"""Self Expanding Hierarchical MSM base class"""

# Author: Kessem Clein <kessem.clein@mail.huji.ac.il>

from abc import ABC

import warnings
import numpy as np
from HMSM import HMSMConfig, models
from HMSM.util import util
from HMSM.samplers.trajectory_samplers import BaseTrajectorySampler
from HMSM.discretizers import BaseDiscretizer

def _final_states(dtrajs):
    return [dtraj[-1] for dtraj in dtrajs]

class SelfExpandingHierarchicalMSM(ABC):
    """SelfExpandingHierarchicalMSM
    This class is used to build and manage a hierarchical MSM over some energy landscape, as will
    be described in Clein et al. 2021. It relies on the provided sampler to explore microstates,
    that are stored at the vertices of lowest (highest-resolution) level in the MSM. It then creates
    a hierarchical (multiscale) map of the states in the configuration space, and stores it in a
    HierarchicalMSMTree data structure.

    Parameters
    ----------
    tree_type : HierarchicalMSMTreeBase
        The type of HMSM tree to use for this model.
    sampler : BaseTrajectorySampler
        A wrapper for some process that samples sequences from a Markov-chain.
    discretizer : BaseDiscretizer

    sampling_optimizer_type : str
        The type of optimizer to use for sampling
    partition_estimator_type : {'auto', 'manual'}, str
        The type of optimizer to use for partitioning MSMs.
    partition_estimator : MetastablePartition
        If partition_estimator is 'manual', this argument will be used as the partition
        optimizer.



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
    >>> sampler = BrownianDynamicsSampler(force_function, dim=2, dt=2e-15, kT=1)
    >>> discretizer = KCentersCoarseGrain(cutoff=3, dim=2)
    >>> hmsm = SelfExpandingHierarchicalMSM(sampler, discretizer)

    sample and update in batches for 1 minute in CPU time:

    >>> tree = hmsm.expand(max_cputime=60)

    sample and update in batches for 1 minute in CPU time, using the exploration optimizer:

    >>> tree = hmsm.expand(max_cputime=60, optimizer='exploration')

    Now we can do some analysis, visualizations, etc, on the tree:

    TODO: provide example

    Continue sampling until the timescale of the longest process described by the HMSM is at least
    1 second:

    >>> hmsm.expand(min_timescale_sec=1)         # tree points to the HierarchicalMSMs tree data
    >>> timescale = tree.get_longest_timescale() # structure, so it was updated by hmsm.expand
    >>> timescale_in_seconds = timescale * hmsm.timestep_in_seconds
    >>> print(f"The timescale of the slowest process described by this HMSM is \
                {timescale_in_seconds:.1f} seconds")
    The timescale of the slowest process described by this HMSM is 1.2 seconds
    """


    def __init__(self, sampler:BaseTrajectorySampler, discretizer:BaseDiscretizer,
                 config:HMSMConfig=None, **config_kwargs):
        self._sampler = sampler
        self._discretizer = discretizer
        if config is None:
            self.config = HMSMConfig(**config_kwargs)
        else:
            self.config = config
        self._hmsm_tree = self._init_tree()
        self._n_samples = self._last_force_rebuild = 0
        self._effective_timestep_seconds = self._sampler.dt * self.config.base_tau
        self._init_sample()
        self._rebuild_interval = 1e6

    def _init_tree(self):
        if self.config.tree_type in ('auto', 'serial'):
            tree = models.serial.HierarchicalMSMTree(self.config)
        else:
            raise NotImplementedError(f"tree_type {self.config.tree_type}, not implemented, only \
                                        'serial' is currently supported.")
        return tree

    def _init_sample(self):
        dtrajs = self._sampler.get_initial_sample(self.config.trajectory_len,
                                                  self.config.n_trajectories,
                                                  self.config.base_tau)
        self._hmsm_tree.update_model_from_trajectories(dtrajs)
        self._equilibrium_sample = _final_states(dtrajs)
        self._n_samples += self.config.n_trajectories * self.config.trajectory_len

    def _get_equilibrium_sample(self, dtrajs):
        states = _final_states(dtrajs)
        restart_states = np.random.binomial(1, 
                                            self.config.restart_fraction,
                                            self.config.n_trajectories)
        new_states = self.tree.sample_states(np.sum(restart_states))

        for i, restart in enumerate(restart_states):
            if restart:
                states[i] = new_states.pop()
        return states


    @property
    def batch_size(self):
        return self.config.n_microstates * self.config.n_samples * self.config.sample_len + \
               self.config.n_trajectories * self.config.trajectory_len

    @property
    def timestep_in_seconds(self):
        return self._effective_timestep_seconds

    @property
    def tree(self):
        return self._hmsm_tree

    @property
    def discretizer(self):
        return self._discretizer

    def get_timescale(self):
        #TODO document or eliminate
        return self._hmsm_tree.get_longest_timescale() * self.timestep_in_seconds

    def expand(self, max_cputime=np.inf, max_samples=np.inf):
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

        if max_cputime == max_samples == np.inf:
            # TODO: throw exception instead
            warnings.warn("At least one of the parameters max_cputime, max_samples, or \
                              min_timescale_sec must be given")
            return

        max_values = {'n_samples' : max_samples}
        stop_condition = util.get_threshold_check_function(max_values, max_time=max_cputime)
        n_samples = 0
        timescale = np.inf
        batch_size = self.batch_size

        # Main loop: 1) sample from equilibrium 2) adaptive sampling for refinement 3) update; repeat
        while not stop_condition(n_samples=n_samples):
            # equilibrium sampling:
            equilibrium_dtrajs = self._sampler.sample_from_states(self._equilibrium_sample,
                                                                  self.config.trajectory_len,
                                                                  1,
                                                                  self.config.base_tau)

            # choose microstates for adaptive sampling, and get trajectories:
            microstates = self._hmsm_tree.sample_states(n_samples=self.config.n_microstates)
            adaptive_dtrajs = self._sampler.sample_from_states(microstates,
                                                          self.config.sample_len,
                                                          self.config.n_samples,
                                                          self.config.base_tau)

            self._hmsm_tree.update_model_from_trajectories(adaptive_dtrajs + equilibrium_dtrajs)
            self._equilibrium_sample = self._get_equilibrium_sample(equilibrium_dtrajs)

            # some book keeping:
            n_samples += batch_size
            self._n_samples += batch_size
            if self._n_samples - self._last_force_rebuild >= self._rebuild_interval:
                self._hmsm_tree.force_rebuild_tree()
                self._last_force_rebuild = self._n_samples
                self._rebuild_interval *= 2

