
from abc import ABC

class MetastablePartition(ABC):

    @abstractmethod
    def check_split_condition(self, *args, **kwargs) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get_metastable_partition(self, *args, **kwargs):
        """get_metastable_partition.

        Returns
        -------
        partition : List[Iterable]        
            A partition of states.
        taus : List[int]
            The taus (timestep resolutions) of each element of the partition.
        """
        raise NotImplementedError()
