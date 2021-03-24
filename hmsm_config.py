"""HMSMConfig"""

# Author: Kessem Clein <kessem.clein@mail.huji.ac.il>

from dataclasses import dataclass, field

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
    parent_update_condition: str='auto'
    parent_update_threshold: float=0.1
    sampling_optimizer: str='auto'
    sampling_optimizer_kwargs: dict=field(default_factory=dict)
    partition_estimator: str='auto'
    partition_kwargs: dict=field(default_factory=dict)
    transition_estimator: str='Dirichlet_MMSE'
    """
    base_tau: int=1
    n_microstates: int=20
    n_samples: int=5
    sample_len: int=5
    parent_update_condition: str='auto'
    parent_update_threshold: float=0.1
    sampling_optimizer: str='auto'
    sampling_optimizer_kwargs: dict=field(default_factory=dict)
    partition_estimator: str='auto'
    partition_kwargs: dict=field(default_factory=dict)
    transition_estimator: str='Dirichlet_MMSE'
