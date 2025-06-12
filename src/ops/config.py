"""
Typed configuration objects for the Thermur project, managed by Pydantic and
registered with hydra-zen for command-line accessibility.

This module defines a hierarchical structure for all parameters, from the
environment and swarm dynamics to the GNN policy and training loop. This
ensures that all components are configured from a single, validated source of
truth.
"""
from hydra_zen import store, ZenField
from pydantic  import BaseModel, Field
from typing    import Literal


# --------------------------------------------------------------------------
# Sub-Configurations (Sorted Alphabetically)
# --------------------------------------------------------------------------

class AgentConfig(BaseModel, extra="forbid"):
    """
    Defines the physical and sensory properties of a single agent.

    This class consolidates parameters that govern an agent's thermal
    survivability and its function as a perceptible information display. The
    thermal properties are critical for modeling the agent's internal state
    and enforcing safety guarantees.

    The `max_temperature` provides Tₘₐₓ for the Control Barrier Function's
    safety boundary: h(𝐱) = Tₘₐₓ - T(𝐱). The `thermal_time_constant` (τ) is
    used in the agent's internal RC thermal model to estimate core temperature
    from skin temperature: T_core ≈ T_skin - τ ⋅ dT_skin/dt.
    """
    led_color_space: Literal["CIELAB", "Oklab"] = Field(
        default     = "CIELAB",
        description = (
            "The perceptually-uniform color space for mapping temperature to a "
            "visible color."
        )
    )
    max_temperature: float = Field(
        default     = 500.0,
        gt          = 0,
        description = (
            "Maximum survivable agent temperature in Fahrenheit (°F), defining "
            "the hard safety boundary h(𝐱) for the CBF."
        )
    )
    thermal_time_constant: float = Field(
        default     = 5.0,
        gt          = 0,
        description = (
            "RC thermal model time constant (τ) in seconds, used to estimate "
            "internal temperature."
        )
    )


class CBFConfig(BaseModel, extra="forbid"):
    """
    Parameters for the Control Barrier Function (CBF) safety filter.

    The CBF guarantees forward invariance of the safety set C, ensuring agents
    do not exceed `max_temperature` via the constraint: ḣ(𝐱) ≥ -αh(𝐱). This
    is solved in real-time via a Quadratic Program (QP).
    """
    alpha: float = Field(
        default     = 0.5,
        gt          = 0,
        description = (
            "Class-K function gain (α) for the CBF safety constraint, "
            "controlling convergence to the safe set."
        )
    )


class EnvironmentConfig(BaseModel, extra="forbid"):
    """
    Configuration for the simulation environment.

    This class specifies the environment to be instantiated, including its
    dynamics and the source of the physical data (wind and temperature fields)
    that it will provide to the agents.
    """
    data_source: str = Field(
        default     = "data/wrfout_d01.nc",
        description = (
            "Path to the environmental data source (e.g., NetCDF from "
            "WRF-Fire)."
        )
    )
    name: str = Field(
        default     = "WRF-Fire-Env-v0",
        description = "The registered name of the Gymnasium environment to use."
    )
    simulation_step: float = Field(
        default     = 0.05,
        gt          = 0,
        description = (
            "The duration of a single simulation physics step (Δt) in seconds."
        )
    )


class ExpertPolicyConfig(BaseModel, extra="forbid"):
    """
    Defines the weights for the handcrafted 'expert' flocking controller.

    This controller's nominal action, 𝐮_nom, is derived from the negative
    gradient of a synthetic potential energy function, U = -∇ₓU(Sₜ). These
    parameters weight the components of that function, which are based on
    classic Reynolds rules and our thermal constraints.
    - Cohesion   : U_coh   ∝ Σ||xᵢ - xⱼ||²
    - Separation : U_sep   ∝ Σ 1/||xᵢ - xⱼ||
    - Alignment  : U_align ∝ Σ||vᵢ - vⱼ||²
    - Thermal    : U_therm ∝ 1/(Tₘₐₓ - Tᵢ)
    """
    w_alignment: float = Field(
        default     = 0.8,
        description = (
            "Weight for the alignment potential. Higher values encourage agents "
            "to match velocity with neighbors."
        )
    )
    w_cohesion: float = Field(
        default     = 1.0,
        description = (
            "Weight for the cohesion potential. Higher values encourage agents "
            "to form a tighter group."
        )
    )
    w_separation: float = Field(
        default     = 1.5,
        description = (
            "Weight for the separation potential. Higher values create more "
            "space between nearby agents."
        )
    )
    w_thermal: float = Field(
        default     = 2.0,
        description = (
            "Weight for the thermal potential. Higher values create a stronger "
            "repulsion from high-temperature regions."
        )
    )


class GNNConfig(BaseModel, extra="forbid"):
    """
    Defines the architecture of the Graph Neural Network (GNN) policy, π_θ.

    This policy is trained to imitate the expert controller. At each step, it
    performs message passing where each node aggregates features from its
    neighbors (𝐚ᵢ = 𝚺 hⱼ) and updates its own hidden state (hᵢ' = GRU(hᵢ, 𝐚ᵢ)).
    """
    activation: Literal["relu", "silu", "tanh"] = Field(
        default     = "silu",
        description = (
            "The nonlinearity used in the GNN's multi-layer perceptrons (MLPs)."
        )
    )
    hidden_dim: int = Field(
        default     = 64,
        gt          = 0,
        description = "Dimensionality of the hidden node embeddings and messages.",
    )
    num_layers: int = Field(
        default     = 3,
        ge          = 1,
        description = (
            "Number of GNN message-passing layers. More layers increase the "
            "agent's receptive field but also computational cost."
        )
    )


class LoggingConfig(BaseModel, extra="forbid"):
    """
    Configuration for the Loguru logging setup.

    This class controls the verbosity, format, and destinations of log
    messages generated throughout the application.
    """
    colorize: bool = Field(
        default     = True,
        description = "Whether to use colorized log output for the console."
    )
    diagnose: bool = Field(
        default     = False,
        description = "Whether to add exception tracebacks to the log for debugging."
    )
    enqueue: bool = Field(
        default     = True,
        description = "Whether to make file logging non-blocking and thread-safe."
    )
    file_path: str | None = Field(
        default     = "logs/thermur.log",
        description = "Path to the log file. If None, file logging is disabled."
    )
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default     = "INFO",
        description = "The minimum log level to be processed and displayed."
    )
    retention: str = Field(
        default     = "7 days",
        description = "Log file retention policy (e.g., '10 days', '1 month')."
    )
    rotation: str = Field(
        default     = "10 MB",
        description = "Log file rotation policy (e.g., '500 MB', '12:00')."
    )


class PolicyConfig(BaseModel, extra="forbid"):
    """
    Groups all configurations related to an agent's decision-making process.

    This class orchestrates the complete control pipeline, from high-level
    policy to low-level safety guarantees. The pipeline operates as follows:
    1. The GNN policy (`gnn`) produces a desired nominal action, 𝐮_nom.
    2. The Safety Filter, parameterized by `cbf`, solves a QP to transform
       𝐮_nom into a final, safe action 𝐮*.

    The `expert` configuration is used during the pre-training phase to
    generate the behavioral cloning dataset.
    """
    cbf    : CBFConfig          = CBFConfig()
    expert : ExpertPolicyConfig = ExpertPolicyConfig()
    gnn    : GNNConfig          = GNNConfig()


class SwarmConfig(BaseModel, extra="forbid"):
    """
    Configures the collective properties and initial state of the agent swarm.

    These parameters define the scale of the multi-agent system and the rules
    for local interaction. The `communication_range` is particularly critical
    as it defines the dynamic graph topology Gₜ = (V, Eₜ) at each timestep.
    This metric-based neighborhood is a practical starting point, while natural
    flocks often use a fixed topological neighborhood (e.g., 6-7 nearest agents).
    """
    agent_count: int = Field(
        default     = 30,
        gt          = 1,
        description = "The number of agents (N) in the swarm."
    )
    communication_range: float = Field(
        default     = 50.0,
        gt          = 0,
        description = (
            "The metric distance in meters for defining the topological "
            "neighborhood graph."
        )
    )
    initial_formation: Literal["sphere", "cube"] = Field(
        default     = "sphere",
        description = (
            "The geometric formation of the swarm at the start of the simulation."
        )
    )
    spatial_dims: int = Field(
        default     = 3,
        gt          = 1,
        description = (
            "The number of spatial dimensions in the simulation (e.g., 2 for 2D, 3 "
            "for 3D)."
        )
    )


class TrainConfig(BaseModel, extra="forbid"):
    """
    Parameters for the training and optimization loop.

    These settings govern the imitation learning process (behavioral cloning),
    including the optimizer, batching, and total training duration. The loss
    function will minimize the Mean Squared Error (MSE) between the GNN's
    output and the expert's actions: L_imitation = ||π_θ(s) - 𝐮_nom||².
    """
    batch_size: int = Field(
        default     = 256,
        description = "The number of agent experiences per training batch."
    )
    device: str = Field(
        default     = "cpu",
        description = "The compute device ('cpu', 'cuda', 'mps') for training."
    )
    learning_rate: float = Field(
        default     = 3e-4,
        description = "The learning rate for the AdamW optimizer."
    )
    log_interval: int = Field(
        default     = 1000,
        description = "The frequency (in timesteps) at which to log training metrics."
    )
    seed: int = Field(
        default     = 42,
        description = "The global random seed for ensuring reproducibility."
    )
    total_timesteps: int = Field(
        default     = 200_000,
        description = "The total number of environment steps for the training run."
    )


class WandbConfig(BaseModel, extra="forbid"):
    """
    Configuration for Weights & Biases experiment tracking.

    These parameters control how training runs are logged and organized for
    visualization, analysis, and comparison.
    """
    entity: str | None = Field(
        default     = None,
        description = "The W&B entity (username or team name)."
    )
    mode: Literal["online", "offline", "disabled"] = Field(
        default     = "online",
        description = "The W&B run mode."
    )
    project: str = Field(
        default     = "thermur",
        description = "The W&B project name to log runs into."
    )


# --------------------------------------------------------------------------
# Main Application Configuration
# --------------------------------------------------------------------------
# We use hydra-zen's `make_config` to assemble our Pydantic models into a
# single, cohesive configuration tree that Hydra can manage. This AppConfig
# serves as the single source of truth for an entire application run.

AppConfig = store.make_config(
    "AppConfig",
    zen_meta = {
        "doc": (
            "Root configuration for the Thermur application, assembling all "
            "components."
        )
    },

    # Core Domain Components
    agent = ZenField(
        default_factory = AgentConfig,
        doc             = "Physical properties of a single agent"
    ),
    environment = ZenField(
        default_factory = EnvironmentConfig,
        doc             = "Simulation environment parameters"
    ),
    policy = ZenField(
        default_factory = PolicyConfig,
        doc             = "Agent decision-making and safety pipeline"
    ),
    swarm = ZenField(
        default_factory = SwarmConfig,
        doc             = "Collective properties of the agent swarm"
    ),

    # Infrastructure & Training Components
    logging = ZenField(
        default_factory = LoggingConfig,
        doc             = "Application-wide logging setup"
    ),
    train = ZenField(
        default_factory = TrainConfig,
        doc             = "Training loop and optimizer settings"
    ),
    wandb = ZenField(
        default_factory = WandbConfig,
        doc             = "Weights & Biases experiment tracking"
    ),

    # Hydra Boilerplate
    defaults = ["_self_"]
)


# --------------------------------------------------------------------------
# Register Configs with the Hydra Store
# --------------------------------------------------------------------------
# These calls make our AppConfig discoverable by Hydra's command-line
# interface and @hydra.main decorator, enabling easy configuration management.

# Register a base config for general use.
store(
    AppConfig,
    name    = "base_config",
    group   = "thermur_config",
    package = "_global_"
)

# Register a config specifically for the `train` script's config group.
store(
    AppConfig,
    name    = "train_app",
    group   = "train",
    package = "train"
)
