# Imitation Learning Configuration

This directory contains the Hydra-based configuration workload for imitation learning in Thermur. The configuration system orchestrates all components needed for training the GNN policy via behavioral cloning from the expert flocking controller.

## Architecture Overview

The imitation learning configuration follows a three-tier architecture:

1. **Schemas** (`configs/imitation/schemas/`): Pydantic models defining configuration structure and validation
2. **Factories** (`configs/imitation/factories/`): Hydra-zen builders that create component configurations
3. **Workload** (`imitation.py`): Top-level composition using `make_config()`

## Configuration Structure

```python
imitation_config = make_config(
    # Parameter models - validated configuration data
    agent           = builds(AgentModel),
    environment     = builds(EnvironmentModel),
    hyperparameters = builds(HyperparameterModel),
    swarm           = builds(SwarmModel),
    
    # Component builders - instantiation recipes
    data_collector    = build_collector,
    experience_buffer = build_replay_buffer,
    expert_policy     = build_flocking_controller,
    loss_function     = build_loss,
    monitoring        = build_monitoring,
    optimizer         = build_optimizer,
    policy            = build_policy,
    simulation        = build_simulation,
    visualizer        = build_visualizer,
)
```

## Key Components

### Environment & Simulation
- **SimulationEnv**: MuJoCo-based drone swarm environment
- **EnvironmentDataSource**: Thermal data loading and interpolation
- **Edge Index Computation**: Dynamic graph topology based on communication range

### Policies
- **Expert Policy**: Reynolds flocking controller with thermal awareness
- **Learning Policy**: Graph Neural Network (GNN) for imitation learning
- **Safety Filter**: Control Barrier Functions (CBF) for safety constraints

### Training Infrastructure
- **Data Collector**: TorchRL's SyncDataCollector for experience gathering
- **Replay Buffer**: Experience storage with LazyTensorStorage
- **Loss Function**: Behavioral cloning loss
- **Optimizer**: AdamW with configurable learning rate and weight decay

### Monitoring & Visualization
- **Loguru**: Structured logging with configurable levels
- **Weights & Biases**: Experiment tracking and metrics
- **Visualizer**: Real-time swarm behavior rendering

## Usage

The configuration is used by the training command:

```python
from configs   import imitation_config, register_configs
from hydra_zen import instantiate, zen

# Register configurations with Hydra
register_configs()

# Create Hydra-decorated training function
@zen(imitation_config).hydra_main(
    config_name  = "train",
    version_base = None,
)
def train(cfg):
    # Instantiate all components
    components = {
        "environment"   : instantiate(cfg.simulation),
        "expert_policy" : instantiate(cfg.expert_policy),
        "policy"        : instantiate(cfg.policy),
        # ... etc
    }
    
    # Run training
    train_imitation_learning(**components)
```

## Configuration Overrides

Hydra enables flexible configuration overrides via command line:

```bash
# Change learning rate
thermur train hyperparameters.learning_rate=0.001

# Use different swarm size
thermur train swarm.agent_count=50

# Change environment parameters
thermur train environment.simulation_step=0.02

# Load custom experiment
thermur train +experiment=large_swarm
```

## Configuration Hierarchy

- **Parameter Models**: Define the data (what values to use)
  - `AgentModel`: Drone physical properties
  - `SwarmModel`: Multi-agent system configuration
  - `EnvironmentModel`: Simulation settings
  - `HyperparameterModel`: Training parameters

- **Component Builders**: Define the construction (how to build objects)
  - Each builder uses structured interpolation (`SI`) to reference parameters
  - Builders compose multiple configurations into instantiatable objects

## Extension Points

To add new components:

1. Create a schema in `schemas/` with Pydantic validation
2. Create a factory in `factories/` using `builds()`
3. Add to the workload in `imitation.py`
4. Register any new configuration groups

The configuration system is designed to grow with the project while maintaining clarity and type safety.