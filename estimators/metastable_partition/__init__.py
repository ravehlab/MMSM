from .metastable_partition import MetastablePartition
from .gibbs import GibbsPartition
from . import util
from HMSM import HMSMConfig

def get_metastable_partition(config:HMSMConfig):
    """get_metastable_partition.
    Get a metastable partition object derived from MetastablePartition, from parameters of
    an HMSMConfig object.

    Parameters
    ----------
    config : HMSMConfig
        config object

    Returns
    -------
    partition_estimator : inherits MetastablePartition
    """
    if config.partition_estimator in ('auto' or 'Gibbs'):
        return GibbsPartition(**config.partition_kwargs)
    else:
        raise NotImplementedError(f"partition estimator {config.partition_estimator}\
                                    not implemented, only Gibbs is currently supported")

