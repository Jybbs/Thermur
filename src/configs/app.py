"""
Hydra-zen configuration builders for the Thermur application.

This module uses hydra-zen's `builds` function to create instantiable
configurations for all major components. These configurations serve as
"recipes" that Hydra will use to automatically instantiate objects at runtime.
"""
import torch

from hydra_zen                      import builds, make_config, store, ZenField
from hydra_zen.third_party.pydantic import pydantic_parser
from omegaconf                      import SI
from src.configs.pydantic           import *
from src.envs.thermur               import ThermurEnv
from src.models.gnn_policy          import GNNPolicy
from src.physics.potentials         import ExpertFlockingController
from src.physics.safety             import SafetyFilter
from src.scripts.train              import ImitationLoss, TrainingOrchestrator
from torch.optim                    import AdamW
from torchrl.collectors             import SyncDataCollector
from torchrl.data                   import TensorDictReplayBuffer
from torchrl.envs                   import EnvBase
from torchrl.modules                import SafeModule
from typing                         import Type


# --------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------

def calculate_gnn_input_dim(env: EnvBase) -> int:
    """
    Calculates the GNN's input feature dimension dynamically.

    This function introspects the environment's observation specification to
    determine the total number of features for a single agent.

    Args:
        env: The instantiated environment.

    Returns:
        The integer size of the concatenated node feature vector.
    """
    spec = env.observation_spec

    # These are the keys that will be concatenated into the 'x' feature matrix
    feature_keys = [
        "position", "velocity", "temperature", "temperature_grad", "battery"
    ]
    
    total_dims = sum(
        spec[key].shape[-1] for key in feature_keys if key in spec.keys()
    )
    
    return total_dims


# --------------------------------------------------------------------------
# Factory Functions
# --------------------------------------------------------------------------

def create_collector(
    env              : EnvBase,
    policy           : SafeModule,
    total_frames     : int,
    frames_per_batch : int,
    device           : str,
) -> SyncDataCollector:
    """
    Factory function to create the data collector.

    Args:
        env              : The environment instance.
        policy           : The expert policy.
        total_frames     : Total frames to collect.
        frames_per_batch : Frames per batch.
        device           : The device.

    Returns:
        A configured SyncDataCollector.
    """
    return SyncDataCollector(
        create_env_fn    = lambda: env,  # Wrap existing env
        policy           = policy,
        total_frames     = total_frames,
        frames_per_batch = frames_per_batch,
        device           = torch.device(device),
    )


def create_expert_policy(
    controller : Type,  # ExpertFlockingController
    env        : EnvBase,
    device     : str,
) -> SafeModule:
    """
    Factory function to create the expert policy SafeModule.

    Args:
        controller : The instantiated expert controller.
        env        : The environment (used for action spec).
        device     : The device to place the module on.

    Returns:
        A SafeModule wrapping the expert controller.
    """
    return SafeModule(
        module   = controller.compute_nominal_action,
        in_keys  = ["observation"],
        out_keys = ["action_expert"],
        spec     = env.action_spec,
    ).to(torch.device(device))


def create_gnn_policy(
    gnn_config   : GNNConfig,
    swarm_config : SwarmConfig,
    env          : EnvBase,
    device       : str,
) -> GNNPolicy:
    """
    Factory function to create the GNN policy.

    Args:
        gnn_config   : The GNN configuration.
        swarm_config : The swarm configuration (for output dim).
        env          : The environment (for input dim calculation).
        device       : The device to place the model on.

    Returns:
        An instantiated GNN policy.
    """
    in_dim = calculate_gnn_input_dim(env)
    
    return GNNPolicy(
        in_dim  = in_dim,
        out_dim = swarm_config.spatial_dims,
        config  = gnn_config
    ).to(torch.device(device))


def create_optimizer(
    loss_module   : ImitationLoss,
    learning_rate : float,
    weight_decay  : float,
) -> AdamW:
    """
    Factory function to create the AdamW optimizer.

    Args:
        loss_module   : The loss module containing parameters to optimize.
        learning_rate : The learning rate.
        weight_decay  : The weight decay (L2 penalty).

    Returns:
        A configured AdamW optimizer.
    """
    return AdamW(
        loss_module.parameters(),
        lr           = learning_rate,
        weight_decay = weight_decay,
    )


def create_replay_buffer(
    batch_size  : int,
    buffer_size : int,
    prefetch    : int,
) -> TensorDictReplayBuffer:
    """
    Factory function to create the replay buffer.

    Args:
        batch_size  : The batch size for sampling.
        buffer_size : The maximum buffer size.
        prefetch    : Number of batches to prefetch.

    Returns:
        A configured TensorDictReplayBuffer.
    """
    return TensorDictReplayBuffer(
        storage     = "memory",
        batch_size  = batch_size,
        buffer_size = buffer_size,
        prefetch    = prefetch,
    )


# --------------------------------------------------------------------------
# Configuration Building Function
# --------------------------------------------------------------------------

def build_app_config():
    """
    Builds and returns the main application configuration.
    
    This function is called lazily to avoid circular imports at module load time.
    """
    
    # Environment Configuration
    ThermurEnvConf = builds(
        ThermurEnv,
        config                  = SI("${config}"),  # Will receive the full AppConfig
        populate_full_signature = True,
        zen_dataclass           = {
            "module"   : "configs.app",
            "cls_name" : "ThermurEnvConf"
        }
    )
    
    # Policy Configurations
    ExpertFlockingControllerConf = builds(
        ExpertFlockingController,
        expert_config = SI("${config.policy.expert}"),
        agent_config  = SI("${config.agent}"),
        populate_full_signature = True,
        zen_dataclass = {
            "module"  : "configs.app",
            "cls_name" : "ExpertFlockingControllerConf"
        }
    )
    
    ExpertPolicyConf = builds(
        create_expert_policy,
        controller = SI("${expert_controller}"),
        env        = SI("${env}"),
        device     = SI("${config.train.device}"),
        populate_full_signature = True,
        zen_dataclass = {
            "module"   : "configs.app",
            "cls_name" : "ExpertPolicyConf"
        }
    )
    
    GNNPolicyConf = builds(
        create_gnn_policy,
        gnn_config   = SI("${config.policy.gnn}"),
        swarm_config = SI("${config.swarm}"),
        env          = SI("${env}"),
        device       = SI("${config.train.device}"),
        populate_full_signature = True,
        zen_dataclass = {
            "module"   : "configs.app",
            "cls_name" : "GNNPolicyConf"
        }
    )
    
    # Safety Filter Configuration
    SafetyFilterConf = builds(
        SafetyFilter,
        config = SI("${config.safety}"),
        populate_full_signature = True,
        zen_dataclass = {
            "module"   : "configs.app",
            "cls_name" : "SafetyFilterConf"
        }
    )
    
    # Data Pipeline Configurations
    CollectorConf = builds(
        create_collector,
        env              = SI("${env}"),
        policy           = SI("${expert_policy}"),
        total_frames     = SI("${config.train.collector.total_frames}"),
        frames_per_batch = SI("${config.train.collector.frames_per_batch}"),
        device           = SI("${config.train.device}"),
        populate_full_signature = True,
        zen_dataclass = {
            "module"   : "configs.app",
            "cls_name" : "CollectorConf"
        }
    )
    
    ReplayBufferConf = builds(
        create_replay_buffer,
        batch_size  = SI("${config.train.replay.batch_size}"),
        buffer_size = SI("${config.train.replay.buffer_size}"),
        prefetch    = SI("${config.train.replay.prefetch}"),
        populate_full_signature = True,
        zen_dataclass = {
            "module"   : "configs.app",
            "cls_name" : "ReplayBufferConf"
        }
    )
    
    # Loss and Optimizer Configurations
    ImitationLossConf = builds(
        ImitationLoss,
        policy_network = SI("${gnn_policy}"),
        populate_full_signature = True,
        zen_dataclass = {
            "module"   : "configs.app",
            "cls_name" : "ImitationLossConf"
        }
    )
    
    OptimizerConf = builds(
        create_optimizer,
        loss_module   = SI("${loss_module}"),
        learning_rate = SI("${config.train.learning_rate}"),
        weight_decay  = SI("${config.train.weight_decay}"),
        populate_full_signature = True,
        zen_dataclass = {
            "module"   : "configs.app",
            "cls_name" : "OptimizerConf"
        }
    )
    
    # Training Orchestrator Configuration
    TrainingOrchestratorConf = builds(
        TrainingOrchestrator,
        env           = SI("${env}"),
        expert_policy = SI("${expert_policy}"),
        policy        = SI("${gnn_policy}"),
        collector     = SI("${collector}"),
        replay_buffer = SI("${replay_buffer}"),
        loss_module   = SI("${loss_module}"),
        optimizer     = SI("${optimizer}"),
        train_config  = SI("${config.train}"),
        wandb_config  = SI("${config.wandb}"),
        populate_full_signature = True,
        zen_dataclass = {
            "module"   : "configs.app",
            "cls_name" : "TrainingOrchestratorConf"
        }
    )
    
    # Create the main config that assembles all components
    return make_config(
        # Pydantic configurations
        config = make_config(
            agent       = AgentConfig(),
            environment = EnvironmentConfig(),
            logging     = LoggingConfig(),
            policy      = PolicyConfig(),
            safety      = SafetyConfig(),
            swarm       = SwarmConfig(),
            train       = TrainConfig(),
            wandb       = WandbConfig(),
        ),
        
        # Component builds
        collector         = CollectorConf,
        env               = ThermurEnvConf,
        expert_controller = ExpertFlockingControllerConf,
        expert_policy     = ExpertPolicyConf,
        gnn_policy        = GNNPolicyConf,
        loss_module       = ImitationLossConf,
        optimizer         = OptimizerConf,
        replay_buffer     = ReplayBufferConf,
        safety_filter     = SafetyFilterConf,
        
        # The main orchestrator that will be instantiated
        orchestrator = TrainingOrchestratorConf,
        
        # Hydra defaults
        defaults = ["_self_"],
    )


# --------------------------------------------------------------------------
# Lazy Configuration Access
# --------------------------------------------------------------------------

# Cache for the built config
_app_config = None

def get_app_config():
    """
    Returns the application configuration, building it lazily if needed.
    """
    global _app_config
    if _app_config is None:
        _app_config = build_app_config()
    return _app_config


# For backward compatibility
AppConfig = get_app_config()


# --------------------------------------------------------------------------
# Register Configurations with Hydra Store
# --------------------------------------------------------------------------

def register_configs():
    """
    Register configs with Hydra store when needed.
    """
    config = get_app_config()
    
    # Register the main application config
    store(
        config,
        name    = "app",
        group   = "config",
        package = "_global_"
    )
    
    # Also register for the train script specifically
    store(
        config,
        name    = "train",
        group   = "config", 
        package = "_global_"
    )
    
    # Add to hydra store
    store.add_to_hydra_store(overwrite_ok=True)
