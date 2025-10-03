"""
Defines the Graph Neural Network (GNN) policy module, π_θ.

This file contains the implementation of the `th.nn.Module` that serves as
the agent's brain. The policy, denoted π_θ, is a GNN designed to process the
flock's state, which is naturally represented as a dynamic graph. It learns to
output a nominal velocity command, 𝐮_nom, for each agent.

The architecture is explicitly designed to be configurable and to consume
`torch_geometric.data.Data` objects, which are generated from the environment's
PyG Batch observations.
"""
from __future__          import annotations
from .metrics            import BaseMetric
from pytorch_lightning   import LightningModule
from torch               import nn
from torch.nn            import GRUCell, LayerNorm, Linear, ModuleList, Sequential
from torch.nn.functional import mse_loss
from torch_geometric.nn  import GCNConv
from typing              import Callable, TYPE_CHECKING

import torch as th

if TYPE_CHECKING:
    from .metrics                          import MetricsFactory
    from config.imitation.training         import ArchitectureModel
    from config.types                      import FlockBatch
    from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig, STEP_OUTPUT
    from torch                             import Tensor
    from torch.optim                       import Optimizer
    from torch.optim.lr_scheduler          import LRScheduler
    from torch_geometric.typing            import Adj
    from torchmetrics                      import MetricCollection


class GRUBlock(nn.Module):
    """
    Graph Recurrent Unit block combining GCN convolution with GRU recurrence.

    This module encapsulates the processing pattern used in each layer of the
    GNN policy: message passing via graph convolution, followed by recurrent
    state update via GRU, with normalization for training stability.

    The forward pass implements:

        hₗ₊₁ = LN(GRU(σ(GCN(hₗ, E)), hₗ))

    This design provides:
    - Spatial aggregation via GCN for neighbor information
    - Temporal memory via GRU, which stabilizes control over time
    - Output normalization via LayerNorm, which prevents explosion

    Args:
        dim            : Hidden dimension for all layers
        activation     : Activation function class (e.g., nn.ReLU)
        add_self_loops : Whether GCN includes self-loops
        normalize      : Whether to apply LayerNorm to output
    """
    def __init__(
        self,
        dim            : int,
        activation     : type[nn.Module],
        add_self_loops : bool = False,
        normalize      : bool = True
    ):
        super().__init__()
        self.conv       = GCNConv(dim, dim, add_self_loops=add_self_loops)
        self.gru        = GRUCell(dim, dim)
        self.activation = activation()
        self.norm       = LayerNorm(dim) if normalize else nn.Identity()

    def forward(
        self, 
        h          : th.Tensor, 
        edge_index : Adj
    ) -> th.Tensor:
        """
        Process node features through graph convolution and recurrent update.

        Args:
            h          : Node hidden states [num_nodes, dim]
            edge_index : Graph connectivity [2, num_edges]

        Returns:
            Updated node hidden states [num_nodes, dim]
        """
        return self.norm(self.gru(self.activation(self.conv(h, edge_index)), h))


class GNNPolicy(LightningModule):
    """
    A Graph Neural Network policy for multi-agent flocking control.

    This network implements the decentralized control policy described in the
    project's mathematical framework. It is permutation-equivariant, meaning the
    output for an agent does not depend on the ordering of its neighbors.

    The architecture follows a standard Encoder-Processor-Decoder paradigm:
    1.  Encoder: An MLP that projects the raw node features (kinematics,
        thermal readings) into a higher-dimensional latent space.

    2.  Processor: A series of message-passing layers. Each layer consists
        of a GNN convolution to aggregate neighbor information, followed by a
        GRUCell to update the agent's hidden state. The GRU provides temporal
        memory, helping to stabilize the policy's behavior over time.

    3.  Decoder: A final MLP that maps the agent's processed hidden state
        back to a tangible control action (a 3D velocity vector).
    """
    def __init__(
        self,
        architecture : ArchitectureModel,
        metrics      : MetricsFactory,
        optimizer    : Callable[..., Optimizer],
        scheduler    : Callable[..., LRScheduler]
    ):
        """
        Initializes the GNN policy network.

        Args:
            architecture : Configuration for GNN architecture including hidden
                           dimensions, number of layers, activation function, and
                           I/O dimensions.
            metrics      : Factory for creating metric collections.
            optimizer    : Pre-configured optimizer partial from hydra-zen.
            scheduler    : Pre-configured scheduler partial from hydra-zen.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["metrics", "optimizer", "scheduler"])

        dim, n     = architecture.hidden_dim, architecture.num_layers
        activation = getattr(nn, architecture.activation)

        self.encoder   = Sequential(Linear(13, dim), activation(), LayerNorm(dim))
        self.layers    = ModuleList([GRUBlock(dim, activation) for _ in range(n)])
        self.decoder   = Linear(dim, 3)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.t_metrics = metrics.create_training_metrics()
        self.v_metrics = metrics.create_validation_metrics()

    def _step(
        self,
        batch   : FlockBatch,
        metrics : MetricCollection,
        prefix  : str
    ) -> STEP_OUTPUT:
        """
        Shared step logic for training and validation.

        Uses Lightning's default logging behavior: training logs at step-level,
        validation logs at epoch-level. This produces clean metric names without
        suffixes in wandb.

        Args:
            batch   : PyG Batch containing observations and expert actions
            metrics : Metric collection to update
            prefix  : Logging prefix ('training' or 'validation')

        Returns:
            Scalar loss tensor for backpropagation
        """
        predictions = self(batch)
        mse         = mse_loss(predictions, batch.action)

        metrics.update(batch, predictions)
        self.log(
            batch_size = batch.num_graphs,
            name       = f'{prefix}/mse',
            prog_bar   = True,
            sync_dist  = False,
            value      = mse
        )

        self.log_dict(
            batch_size = batch.num_graphs,
            dictionary = metrics.compute(),
            sync_dist  = False
        )

        return mse

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """
        Configures the optimizer and learning rate scheduler for training.

        Lightning calls this method to set up optimizers and learning rate
        schedulers. The optimizer and scheduler are pre-configured partials
        from hydra-zen that just need their final parameters.

        Returns:
            Configuration for optimizer and learning rate scheduler
        """
        optimizer = self.optimizer(params=self.parameters())

        return {
            "optimizer"    : optimizer,
            "lr_scheduler" : {
                "scheduler" : self.scheduler(optimizer=optimizer),
                "monitor"   : "validation/mse"
            }
        }

    @th.compile(fullgraph=True, mode='default')
    def forward(self, batch: FlockBatch) -> Tensor:
        """
        Performs the forward pass through the GNN.

        Args:
            batch: PyG Batch containing the batched graph state of the flock,
                   with `x` (node features) and `edge_index` (connectivity).

        Returns:
            A tensor of shape [B*N, 3] representing the nominal velocity
            command for each agent in the batch.

        The forward pass follows the sequence:
            x  → encoder(Linear → activation → LayerNorm) → h₀
            h₀ → [GCN → activation → GRU → LayerNorm → hₗ]ₗ → decoder → u_nom

        where:
            - x     : Input node features (position, velocity, temperature, etc.)
            - hₗ     : Hidden state after layer l, normalized to prevent explosion
            - l     : Layer index from 1 to num_layers
            - u_nom : Nominal velocity command output
        """
        h = self.encoder(batch.x)

        for layer in self.layers:
            h = layer(h, batch.edge_index)

        return self.decoder(h)

    def on_train_batch_end(
        self,
        outputs   : STEP_OUTPUT,
        batch     : FlockBatch,
        batch_idx : int
    ):
        """
        Clear metric cache after each training batch.

        Prevents memory growth from cached computations in BaseMetric.

        Args:
            outputs   : Training step outputs (loss tensor)
            batch     : Current training batch containing graph data
            batch_idx : Index of current batch
        """
        BaseMetric.clear_cache()

    def on_validation_batch_end(
        self,
        outputs   : STEP_OUTPUT,
        batch     : FlockBatch,
        batch_idx : int
    ):
        """
        Clear metric cache after each validation batch.

        Prevents memory growth from cached computations in BaseMetric.

        Args:
            outputs   : Validation step outputs (loss tensor)
            batch     : Current validation batch containing graph data
            batch_idx : Index of current batch
        """
        BaseMetric.clear_cache()

    def training_step(self, batch: FlockBatch, idx: int) -> STEP_OUTPUT:
        """
        In PyTorch Lightning, the model defines its own training logic. This is
        Lightning's standard pattern, in that the model knows how to train itself,
        eliminating the need for external training loops.

        Args:
            batch : PyG Batch containing graph observations and expert actions
            idx   : Current batch index (automatically provided by Lightning)

        Returns:
            Scalar MSE loss tensor for automatic backpropagation
        """
        return self._step(batch, self.t_metrics, 'training')

    def validation_step(self, batch: FlockBatch, idx: int) -> STEP_OUTPUT:
        """
        Lightning calls this method during validation to assess the model's
        performance on held-out data. This helps monitor generalization and
        detect overfitting during training.

        Args:
            batch : PyG Batch containing validation observations and actions
            idx   : Current batch index (automatically provided by Lightning)

        Returns:
            Scalar validation MSE loss for automatic metric aggregation
        """
        return self._step(batch, self.v_metrics, 'validation')
