"""HMSMConfig"""

# Author: Kessem Clein <kessem.clein@mail.huji.ac.il>

from dataclasses import dataclass

@dataclass
class HMSMConfig:
    """HMSMConfig.

    This class defines all the parameters used by a SelfExpandingHierarchicalMSM.


    #TODO add descriptions of the parameters
    Parameters
    ----------
    base_tau: int=1
    n_microstates: int=20
    n_samples: int=5
    sample_len: int=5
    split_condition: str='auto'
    split_method: str='auto'
    sample_method: str='auto'
    parent_update_condition: str='auto'
    parent_update_threshold: float=0.1
    sampling_optimizer_type: str='auto'
    partition_estimator_type: str='auto'
    partition_estimator: str='auto'
    transition_estimator: str='Dirichlet_MMSE'
    """
    base_tau: int=1
    n_microstates: int=20
    n_samples: int=5
    sample_len: int=5
    split_condition: str='auto'
    split_method: str='auto'
    sample_method: str='auto'
    parent_update_condition: str='auto'
    parent_update_threshold: float=0.1
    sampling_optimizer_type: str='auto'
    partition_estimator_type: str='auto'
    partition_estimator: str='auto'
    transition_estimator: str='Dirichlet_MMSE'
