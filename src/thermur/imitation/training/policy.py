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
from __future__            import annotations
from pytorch_lightning     import LightningModule
from torch                 import compile, nn
from torch.nn              import GRUCell, Linear, ModuleList
from torch.nn.functional   import mse_loss
from torch_geometric.nn    import GCNConv
from typing                import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .metrics                          import MetricsFactory
    from config.imitation.training         import ArchitectureModel
    from config.types                      import ThermurBatch
    from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig, STEP_OUTPUT
    from torch                             import Tensor
    from torch.optim                       import Optimizer
    from torch.optim.lr_scheduler          import LRScheduler


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
        architecture     : ArchitectureModel,
        metrics_factory  : MetricsFactory,
        optimizer        : Callable[..., Optimizer],
        scheduler        : Callable[..., LRScheduler]
    ):
        """
        Initializes the GNN policy network.

        Args:
            architecture     : Configuration for GNN architecture including hidden
                               dimensions, number of layers, activation function, and
                               I/O dimensions.
            metrics_factory  : Factory for creating metric collections.
            optimizer        : Pre-configured optimizer partial from hydra-zen.
            scheduler        : Pre-configured scheduler partial from hydra-zen.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["metrics_factory"])
        
        dim, n = architecture.hidden_dim, architecture.num_layers
        layers = lambda m: ModuleList([m(dim, dim) for _ in range(n)])
        
        self.activation    = getattr(nn, architecture.activation)()
        self.convs         = layers(GCNConv)
        self.decoder       = Linear(dim, 3)
        self.encoder       = Linear(13, dim)
        self.grus          = layers(GRUCell)
        self.optimizer     = optimizer
        self.scheduler     = scheduler
        self.train_metrics = metrics_factory.create_training_metrics()
        self.val_metrics   = metrics_factory.create_validation_metrics()
        
        if architecture.compile:
            self.forward = compile(self.forward, mode="default")

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
                "monitor"   : "validation/loss"
            }
        }

    def forward(self, batch: ThermurBatch) -> Tensor:
        """
        Performs the forward pass through the GNN.

        Args:
            batch: PyG Batch containing the batched graph state of the flock,
                   with `x` (node features) and `edge_index` (connectivity).

        Returns:
            A tensor of shape [B*N, 3] representing the nominal velocity 
            command for each agent in the batch.

        The forward pass follows the sequence:
            x → encoder → h → [GCN → activation → GRU → h]ₗ → decoder → u_nom

        where:
            - x: Input node features (position, velocity, temperature, etc.)
            - h: Hidden state representations of each agent
            - l: Layer index from 1 to num_layers
            - u_nom: Nominal velocity command output
        """
        h = self.activation(self.encoder(batch.x))

        for conv, gru in zip(self.convs, self.grus, strict=True):
            h = gru(self.activation(conv(h, batch.edge_index)), h)

        return self.decoder(h)

    def training_step(self, batch: ThermurBatch, idx: int) -> STEP_OUTPUT:
        """
        Executes a single training step using behavioral cloning loss.

        In PyTorch Lightning, the model defines its own training logic. This is
        Lightning's standard pattern, in that the model knows how to train itself,
        eliminating the need for external training loops.

        Args:
            batch : PyG Batch containing graph observations and expert actions
            idx   : Current batch index (automatically provided by Lightning)

        Returns:
            Scalar loss tensor for automatic backpropagation
        """
        predictions = self(batch)
        loss        = mse_loss(predictions, batch.action)
        
        self.train_metrics.update(predictions, batch.action, batch)
        self.log('training/loss', loss, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=False)
        
        return loss

    def validation_step(self, batch: ThermurBatch, idx: int) -> STEP_OUTPUT:
        """
        Executes validation step for model evaluation.

        Lightning calls this method during validation to assess the model's
        performance on held-out data. This helps monitor generalization and
        detect overfitting during training.

        Args:
            batch : PyG Batch containing validation observations and actions
            idx   : Current batch index (automatically provided by Lightning)

        Returns:
            Scalar validation loss for automatic metric aggregation
        """
        predictions = self(batch)
        loss        = mse_loss(predictions, batch.action)
        
        self.val_metrics.update(predictions, batch.action, batch)
        self.log('validation/loss', loss, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)
        
        return loss
