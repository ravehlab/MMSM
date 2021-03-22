"""Discretizer base class"""

# Author: Kessem Clein <kessem.clein@mail.huji.ac.il>

from abc import ABC

class BaseDiscretizer(ABC):
    """BaseDiscretizer.
    Base class for coarse graining of continuous spaces into discrete states.
    """

    @property
    @abstractmethod
    def n_states(self):
        pass

    @abstractmethod
    def get_coarse_grained_states(self, data : np.ndarray):
        """Get the state ids of data points.

        Parameters
        ----------
        data : ndarray of shape (n_samples, n_features)
            The data points to cluster into discrete states.

        Returns
        -------
        labels : list of int
            A list of length n_samples, such that labels[i] is the label of the point data[i]
        """
        pass

    @abstractmethod
    def sample_from(self, state_id):
        """Get a sample representative from a given state.

        Parameters
        ----------
        state_id : int
            The id of the state to sample a representative from.

        Returns
        -------
        x : np.ndarray of shape (n_features)
            The sampled point
        """
        pass
