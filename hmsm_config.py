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
    alpha: int=1
    parent_update_condition: str='auto'
    parent_update_threshold: float=0.1
    sampling_optimizer: str='auto'
    sampling_optimizer_kwargs: dict=field(default_factory=dict)
    partition_estimator: str='auto'
    partition_kwargs: dict=field(default_factory=dict)
    transition_estimator: str='Dirichlet_MMSE'
    """
    ##### sampling parameters #####
        ## equilibrium sampling
    n_trajectories: int=50
    trajectory_len: int=5000
    restart_fraction: float=0.1
    
        ## adaptive sampling ##
    n_microstates: int=50
    n_samples: int=5
    sample_len: int=10
    sampling_heuristics: list=field(default_factory=lambda : ["flux", "exploration"])
    sampling_heuristic_weights: list=field(default_factory=lambda : [0.9, 0.1])
    vertex_sampler_kwargs: dict=field(default_factory=dict)
    vertex_sampler: str='auto' # defaults to WeightedVertexSampler

    ##### Kinetics estimation parameters #####
    tree_type: str='auto'
    base_tau: int=1
    alpha: float=1. # Dirichlet prior
    transition_estimator: str='Dirichlet_MMSE'
    oom: bool=False
    reversible: bool=True

    ##### Tree structure parameters #####
    parent_update_condition: str='auto'
    parent_update_threshold: float=0.1
    partition_estimator: str='auto'
    partition_kwargs: dict=field(default_factory=dict)
