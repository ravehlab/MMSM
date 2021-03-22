"""HierarchicalMSMTree Base Class"""

# Author: Kessem Clein <kessem.clein@mail.huji.ac.il>

from abc import ABC

class BaseHierarchicalMSMTree(ABC):
    """HierarchicalMSMTree.
    Base class for a tree data structure, representing a Hierarchical MSM.

    Parameters
    ----------
    config : dict
        A dictionary of model parameters, here is a list of parameters used by the model:
            n_states: int
                How many states to sample from in a single batch of samples
            n_samples: int
                How many samples to take from each state in a single batch of samples
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
            optimizer: callable
                A function that takes an HierarchicalMSMVertex, and returns the id of one of its
                children in the tree.
            estimator: {'MMSE', 'MLE', 'MLE_reversible'}, default='MMSE'
                The type of estimator used to estimate transition probabilities. 
                (TODO: Only MMSE is currently implemented)
            alpha: int or float, default=1
                If the estimator is MMSE, alpha will be used as the parameter of the Dirichlet 
                prior for transition probabilities.
    """

    def __init__(self, config):

    @abstractmethod
    def update_model_from_trajectories(self, dtrajs):
        """update_model_from_trajectories.
        Update the Hierarchical MSM with data from observed discrete trajectories.

        Parameters
        ----------
        dtrajs : Iterable[int]
            An iterable of ids of states
        """
        pass

    @abstractmethod
    def sample_states(self, n_samples, from_subtree=None, level=0, optimizer=None):
        """sample_states.
        Get a set of states from this HMSM, ideally chosen such that sampling a random walk from
        this state is expected to increase some objective function.

        Parameters
        ----------
        n_samples : int
            Number of states to sample (with replacement).
        subtree : int, default=None
            If not None, sample states only from the subtree whose root is subtree
        level : int, default=0
            The height of the states to return. This must be lower than the height of subtree.
        optimizer : Optimizer, default=None
            The optimizer to use for sampling. By default (None) the Optimizer specified in
            self.config will be used.

        Returns
        -------
        states : Iterable[int]
            An iterable of state ids.
        """
        pass
