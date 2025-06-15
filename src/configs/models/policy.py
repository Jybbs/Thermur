"""
Policy network models.

This module defines the Pydantic models for the Graph Neural Network
policy architecture parameters.
"""
from pydantic import BaseModel, Field
from typing   import Literal


class GNNModel(BaseModel, extra="forbid"):
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
    input_dim: int = Field(
        default     = 11,  # position(3) + velocity(3) + temperature(1) + temp_grad(3) + energy(1)
        gt          = 0,
        description = "Dimensionality of the input node features (concatenated state vector).",
    )
    output_dim: int = Field(
        default     = 3,
        gt          = 0,
        description = "Dimensionality of the output action (spatial dimensions).",
    )
