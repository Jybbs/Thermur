"""
Typed configuration objects for the Thermur project, managed by Pydantic.

This module defines a hierarchical structure for all parameters, from the
environment and swarm dynamics to the GNN policy and training loop. This
ensures that all components are configured from a single, validated source of
truth.
"""
from pydantic import BaseModel, Field
from typing   import Literal


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


class CheckpointConfig(BaseModel, extra="forbid"):
    """
    Parameters for saving model checkpoints during training.
    """
    interval: int = Field(
        default     = 25_000,
        gt          = 0,
        description = "The frequency (in frames) at which to save a model checkpoint."
    )
    path: str = Field(
        default     = "checkpoints/",
        description = "Directory where model checkpoints will be saved."
    )


class CollectorConfig(BaseModel, extra="forbid"):
    """
    Parameters for the torchrl SyncDataCollector.

    This configures the data collection loop, which interacts with the
    environment to gather agent experiences.
    """
    frames_per_batch: int = Field(
        default     = 1024,
        gt          = 0,
        description = "Number of frames (agent steps) to collect per batch."
    )
    total_frames: int = Field(
        default     = 200_000,
        description = "The total number of environment frames to collect for the training run."
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
    Groups all parameters required for agent decision-making.

    This configuration encompasses both the expert policy used for data
    collection and the GNN policy that learns from it.
    """
    expert : ExpertPolicyConfig = ExpertPolicyConfig()
    gnn    : GNNConfig          = GNNConfig()


class QPSolverConfig(BaseModel, extra="forbid"):
    """
    Parameters for the qpth Quadratic Program (QP) solver.

    These settings control the behavior and numerical precision of the
    differentiable QP solver used in the safety filter.
    """
    eps: float = Field(
        default     = 1e-7,
        gt          = 0,
        description = "Tolerance for constraint satisfaction in the QP solver."
    )
    max_iter: int = Field(
        default     = 20,
        gt          = 0,
        description = "Maximum number of iterations for the QP solver."
    )
    on_failure: Literal["error", "use_nominal"] = Field(
        default     = "error",
        description = (
            "Action to take if the QP solver fails. 'error' raises an "
            "exception, 'use_nominal' falls back to the original unsafe action."
        )
    )


class ReplayBufferConfig(BaseModel, extra="forbid"):
    """
    Parameters for the torchrl TensorDictReplayBuffer.

    This configures the experience replay buffer, which stores transitions
    collected from the environment for training.
    """
    batch_size: int = Field(
        default     = 256,
        gt          = 0,
        description = "The number of agent experiences per training batch."
    )
    buffer_size: int = Field(
        default     = 50_000,
        gt          = 0,
        description = "The maximum number of agent experiences to store in the buffer."
    )
    prefetch: int = Field(
        default     = 8,
        ge          = 0,
        description = "Number of batches to prefetch for training to hide data loading latency."
    )


class SafetyConfig(BaseModel, extra="forbid"):
    """
    Groups all parameters required by the `SafetyFilter`.

    This configuration defines the safety boundary, the behavior of the
    Control Barrier Function (CBF), and the underlying Quadratic Program (QP)
    solver. It consolidates parameters from multiple domains: agent properties
    (`agent`), swarm properties (`swarm`), control theory (`cbf`), and
    numerical optimization (`qp`).
    """
    agent : AgentConfig    = AgentConfig()
    cbf   : CBFConfig      = CBFConfig()
    qp    : QPSolverConfig = QPSolverConfig()
    swarm : SwarmConfig    = SwarmConfig()


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
    output and the expert's actions: 
    
        L_imitation = ||π_θ(s) - 𝐮_nom||².
    """
    # Child Configs
    checkpoint : CheckpointConfig   = CheckpointConfig()
    collector  : CollectorConfig    = CollectorConfig()
    replay     : ReplayBufferConfig = ReplayBufferConfig()

    # Top-Level Parameters (alphabetical)
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
        description = "The frequency (in frames) at which to log training metrics."
    )
    seed: int = Field(
        default     = 42,
        description = "The global random seed for ensuring reproducibility."
    )
    weight_decay: float = Field(
        default     = 1e-5,
        ge          = 0,
        description = "Weight decay (L2 penalty) for the AdamW optimizer."
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
