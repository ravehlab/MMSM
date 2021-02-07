import warnings
import time
import numpy as np
from HMSM.prototype.hierarchical_msm_tree import HierarchicalMSMTree
from HMSM.prototype.hmsm_config import get_default_config

def _get_stop_condition(max_time, max_samples, max_timescale):

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

class HierarchicalMSMEstimator:
    """HierarchicalMSMEstimator.
    This class wraps the HierarchicalMSMTree data structure, its construction and
    estimation from a sampler object.

    Parameters
    ----------
    sampler: DiscreteSampler
        A wrapper for some process that samples from a discrete Markov chain.

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
    >>> sampler = BrownianDynamicsSampler(force_function, dim=2, dt=2e-15, kT=1)
    >>> hmsm_estimator = HierarchicalMSMEstimator(sampler, split_method=two_fold_gibbs_split)

    sample and update in batches for 1 minute:

    >>> tree = hmsm_estimator.estimate(max_time=60)

    now we can do some analysis, visualizations, etc, on the tree

    continue sampling until the timescale of the longest process described by the HMSM is
    1 second (2e15*sampler.dt = 2e15*2e-15 = 1 second):

    >>> hmsm_estimator.estimate(max_timescale=2e15)
    >>> tree.get_longest_timescale(dt=sampler.dt)
    1.52794


    """


    def __init__(self, sampler, **config_kwargs):
        self.sampler = sampler
        self.config = get_default_config()
        self.config.update(config_kwargs)

        self.hmsm_tree = HierarchicalMSMTree(self.config)

    @property
    def batch_size(self):
        return self.config["n_microstates"] * self.config["n_samples"] * self.config["sample_len"]

    def estimate(self, max_time=None, max_samples=None, max_timescale=None):
        """
        Estimate an HMSM by sampling from the sampler.

        Parameters
        ----------
        max_time :
            Maximum time to run in seconds.
        max_samples :
            Maximum number of samples to use.
        max_timescale :
            Maximum timescale of the full HMSM, after which to stop sampling.
        """

        if max_time is max_samples is max_timescale is None:
            warnings.warn("At least one of the parameters max_time, max_samples, or \
                              max_timescale must be given")
            return

        stop_condition = _get_stop_condition(max_time, max_samples, max_timescale)
        n_samples = 0
        timescale = self.hmsm_tree.get_longest_timescale()
        batch_size = self.batch_size

        while not stop_condition(n_samples, time.time(), timescale):
            microstates = self.hmsm_tree.sample_microstate(n_samples=self.config["n_microstates"])
            dtrajs = self.sampler.sample_from_microstates(microstates,\
                                                          self.config["n_samples"],\
                                                          self.config["sample_len"],
                                                          self.config["base_tau"])
            self.hmsm_tree.update_model_from_trajectories(dtrajs)
            n_samples += batch_size
            timescale = self.hmsm_tree.get_longest_timescale()
