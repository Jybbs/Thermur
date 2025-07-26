"""
Policy network and training configuration.

This module defines configuration models for the GNN policy architecture,
optimization settings, hardware configuration, and model checkpointing.
"""
from pathlib  import Path
from pydantic import BaseModel, Field, NonNegativeFloat
from pydantic import NonNegativeInt, PositiveFloat, PositiveInt
from typing   import Literal, Optional


class ArchitectureModel(BaseModel, extra="forbid"):
    """
    GNN architecture configuration.
    
    Defines the neural network structure for processing the dynamic graph 
    G_t = (V, E_t) through message passing layers, where each layer performs:
    
        h_i^(l+1) = GRU(h_i^(l), Σ_j∈N(i) MLP(h_j^(l)))
    
    This allows the policy to learn decentralized control while respecting
    the communication topology.
    """
    activation: Literal["ReLU", "SiLU", "Tanh"] = Field(
        default     = "SiLU",
        description = (
            "Activation function σ(·) applied after linear transformations in message "
            "passing MLPs, with SiLU providing smooth gradients for stable training."
        )
    )
    hidden_dim: PositiveInt = Field(
        default     = 64,
        description = (
            "Hidden state dimensionality d_h for node embeddings h_i^(l) at each "
            "message passing layer, balancing expressiveness against memory usage."
        )
    )
    input_dim: PositiveInt = Field(
        default     = 11,
        description = (
            "Input feature dimensionality d_in = 11 encoding agent state as concatenated "
            "vector [position(3), velocity(3), temperature(1), temp_gradient(3), energy(1)]."
        )
    )
    num_layers: PositiveInt = Field(
        default     = 3,
        description = (
            "Message passing depth L determining multi-hop receptive field radius "
            "r = L × R_comm for aggregating information from distant neighbors."
        )
    )
    output_dim: PositiveInt = Field(
        default     = 3,
        description = (
            "Output action dimensionality d_out matching spatial dimensions for velocity "
            "commands u_i ∈ ℝ^3 sent to low-level flight controllers."
        )
    )


class CheckpointModel(BaseModel, extra="forbid"):
    """
    Model checkpoint configuration.
    
    Controls how and when model checkpoints are saved during training,
    enabling recovery from interruptions and model selection.
    """
    dirpath: Path = Field(
        default     = Path("checkpoints"),
        description = (
            "Directory path for saving model checkpoints during training, enabling "
            "recovery from interruptions and model comparison across runs."
        )
    )
    every_n_train_steps: PositiveInt = Field(
        default     = 25_000,
        description = (
            "Step interval between checkpoint saves, balancing storage costs with "
            "recovery granularity for long training runs on large datasets."
        )
    )
    filename: str = Field(
        default     = "checkpoint-{step}",
        description = (
            "Filename template with {step} placeholder for organizing checkpoints "
            "chronologically, supporting automated model selection pipelines."
        )
    )
    save_last: bool = Field(
        default     = True,
        description = (
            "Always save the final model checkpoint at training completion regardless "
            "of whether it achieved the best validation metric."
        )
    )


class HardwareModel(BaseModel, extra="forbid"):
    """
    Hardware and compute configuration.
    
    Configures the computational resources and strategies used for training,
    including device selection, precision, and distributed training options.
    """
    accelerator: Literal["auto", "cpu", "gpu", "tpu", "mps"] = Field(
        default     = "auto",
        description = (
            "Hardware accelerator type for training computations - automatically "
            "detects and selects the best available device."
        )
    )
    compile_model: bool = Field(
        default     = False,
        description = (
            "Enable PyTorch 2.0 compile for potential speedups. May increase "
            "startup time but can significantly improve training performance."
        )
    )
    devices: PositiveInt = Field(
        default     = 1,
        description = "Number of GPUs/TPUs to use for distributed training."
    )
    precision: Literal["16-mixed", "bf16-mixed", "32-true", "64-true"] = Field(
        default     = "16-mixed",
        description = (
            "Numerical precision mode for training - mixed precision reduces memory "
            "usage and accelerates computation on compatible hardware."
        )
    )
    strategy: Literal["auto", "ddp", "dp", "deepspeed", "fsdp"] = Field(
        default     = "auto",
        description = (
            "Distributed training strategy. 'auto' selects based on hardware, "
            "'ddp' for multi-GPU, 'deepspeed'/'fsdp' for large model training."
        )
    )


class OptimizerModel(BaseModel, extra="forbid"):
    """
    Optimization and learning rate configuration.
    
    Defines the optimization algorithm parameters and learning rate scheduling
    for training the imitation policy to minimize:
    
        L_imitation = ||π_θ(s) - u_nom||²
    
    where π_θ is the learned GNN policy and π* is the expert controller.
    """
    early_stopping_patience: PositiveInt = Field(
        default     = 10,
        description = (
            "Number of epochs without validation improvement before early stopping "
            "triggers to prevent overfitting and save compute."
        )
    )
    gradient_clip_val: PositiveFloat = Field(
        default     = 1.0,
        description = (
            "Maximum gradient norm threshold for clipping to prevent exploding "
            "gradients and maintain stable training dynamics."
        )
    )
    learning_rate: PositiveFloat = Field(
        default     = 3e-4,
        description = (
            "Initial learning rate α for AdamW optimizer, controlling step size "
            "in parameter space during gradient descent optimization."
        )
    )
    lr_factor: PositiveFloat = Field(
        default     = 0.5,
        description = (
            "Factor by which learning rate is reduced when validation loss plateaus. "
            "New LR = old LR × factor."
        )
    )
    lr_patience: PositiveInt = Field(
        default     = 5,
        description = (
            "Number of epochs with no improvement before reducing learning rate. "
            "Works with ReduceLROnPlateau scheduler."
        )
    )
    lr_scheduler_verbose: bool = Field(
        default     = True,
        description = (
            "Whether ReduceLROnPlateau scheduler prints messages when learning "
            "rate is reduced."
        )
    )
    metric: str = Field(
        default     = "train/loss",
        description = (
            "PyTorch Lightning metric key monitored for model selection, early "
            "stopping, and learning rate scheduling decisions during training."
        )
    )
    mode: Literal["min", "max"] = Field(
        default     = "min",
        description = (
            "Optimization direction for monitored metric (minimize or maximize)."
        )
    )
    seed: Optional[NonNegativeInt] = Field(
        default     = 42,
        description = (
            "Random seed for reproducible training runs. Set to None for "
            "non-deterministic behavior."
        )
    )
    weight_decay: NonNegativeFloat = Field(
        default     = 1e-5,
        description = (
            "L2 regularization coefficient λ penalizing large weights to improve "
            "generalization and prevent overfitting to training demonstrations."
        )
    )