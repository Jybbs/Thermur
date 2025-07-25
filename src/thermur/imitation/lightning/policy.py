"""
Defines the Graph Neural Network (GNN) policy module, π_θ.

This file contains the implementation of the `torch.nn.Module` that serves as
the agent's brain. The policy, denoted π_θ, is a GNN designed to process the
flock's state, which is naturally represented as a dynamic graph. It learns to
output a nominal velocity command, 𝐮_nom, for each agent.

The architecture is explicitly designed to be configurable and to consume
`torch_geometric.data.Data` objects, which are generated from the environment's
`TensorDict` observations.
"""
from config.imitation.schemas.learning    import ArchitectureModel, OptimizerModel
from pytorch_lightning                    import LightningModule
from tensordict                           import TensorDict
from thermur.imitation.monitoring.metrics import MetricsCollector
from torch                                import Tensor
from torch.nn                             import GRUCell, Linear, Module, ModuleList
from torch.nn.functional                  import mse_loss
from torch.optim                          import AdamW
from torch.optim.lr_scheduler             import ReduceLROnPlateau
from torch_geometric.data                 import Data
from torch_geometric.nn                   import GCNConv
from typing                               import Type

import torch


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
        metrics      : MetricsCollector,
        optimizer    : OptimizerModel
    ):
        """
        Initializes the GNN policy network.

        Args:
            architecture : Configuration for GNN architecture including hidden 
                           dimensions, number of layers, activation function, and 
                           I/O dimensions.
            metrics      : Centralized metrics collection and management system.
            optimizer    : Configuration for optimization including learning rate,
                           weight decay, and gradient clipping.
        """
        super().__init__()
        self.architecture = architecture
        self.metrics      = metrics
        self.optimizer    = optimizer

        self.save_hyperparameters(ignore=["metrics"])
        self.activation = getattr(torch.nn, architecture.activation)()
        self.convs      = self._build_module_list(architecture, GCNConv)
        self.grus       = self._build_module_list(architecture, GRUCell)
        self.decoder    = Linear(architecture.hidden_dim, architecture.output_dim)
        self.encoder    = Linear(architecture.input_dim,  architecture.hidden_dim)
    
    def _build_module_list(
        self, 
        architecture : ArchitectureModel,
        module_type  : Type[Module]
    ) -> ModuleList:
        """
        Creates a stack of neural network modules of the specified type.
        
        This method unifies the creation of both convolutional and recurrent
        layers, reducing code duplication. For GCN layers, it performs message
        passing: h_i^(l+1) = σ(W^(l) · Σ_j∈N(i) h_j^(l) / |N(i)|). For GRU
        cells, it maintains temporal state with gating mechanisms.
        
        Args:
            module_type : The class of module to instantiate (GCNConv or GRUCell)
            learning    : Configuration containing architecture parameters
            
        Returns:
            ModuleList containing num_layers of the specified module type
        """
        dim = architecture.hidden_dim
        return ModuleList([
            module_type(dim, dim) for _ in range(architecture.num_layers)
        ])
    
    def _compute_loss_and_log(
        self, 
        batch : TensorDict, 
        phase : str
    ) -> Tensor:
        """
        Computes behavioral cloning loss and logs metrics.
        
        This method implements the standard behavioral cloning objective:
        L = MSE(π_θ(s), a*), where π_θ(s) is the policy's predicted action
        and a* is the expert's demonstrated action.
        
        Args:
            batch : TensorDict containing graph observations and expert actions
            phase : Training phase ('train' or 'val') for logging
            
        Returns:
            Scalar loss tensor for backpropagation or metric aggregation
        """
        predictions = self(batch)
        targets     = batch["action"]
        loss        = mse_loss(predictions, targets)
        
        self.metrics.update_imitation_metrics(
            phase       = phase,
            predictions = predictions,
            targets     = targets
        )
        
        self.metrics.log_all_metrics(
            module      = self,
            phase       = phase,
            loss        = loss,
            predictions = predictions,
            targets     = targets
        )
        
        return loss

    def configure_optimizers(self) -> dict:
        """
        Configures the optimizer and learning rate scheduler for training.
        
        Lightning calls this method to set up optimizers and learning rate
        schedulers. Returns the AdamW optimizer with ReduceLROnPlateau
        scheduler that monitors validation loss.
        
        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        optimizer = AdamW(
            params       = self.parameters(),
            lr           = self.optimizer.learning_rate,
            weight_decay = self.optimizer.weight_decay
        )
        
        return {
            "optimizer"    : optimizer,
            "lr_scheduler" : {
                "monitor"   : self.optimizer.metric.replace("train", "val"),
                "scheduler" : ReduceLROnPlateau(
                    factor    = self.optimizer.lr_factor,
                    mode      = self.optimizer.mode,
                    optimizer = optimizer,
                    patience  = self.optimizer.lr_patience,
                    verbose   = self.optimizer.lr_scheduler_verbose
                )
            }
        }

    def forward(self, data: Data) -> Tensor:
        """
        Performs the forward pass through the GNN.

        Args:
            data: A `torch_geometric.data.Data` object containing the batched
                  graph state of the flock, with `x` (node features) and
                  `edge_index` (connectivity).

        Returns:
            A tensor of shape (num_nodes, out_dim) representing the nominal
            velocity command for each agent in the batch.
            
        The forward pass follows the sequence:
            x → encoder → h → [GCN → activation → GRU → h]ₗ → decoder → u_nom
        
        where:
            - x: Input node features (position, velocity, temperature, etc.)
            - h: Hidden state representations of each agent
            - l: Layer index from 1 to num_layers
            - u_nom: Nominal velocity command output
        """
        x, edge_index = data.x, data.edge_index
        h = self.activation(self.encoder(x))
        
        for conv, gru in zip(self.convs, self.grus, strict=True):
            h = gru(self.activation(conv(h, edge_index)), h)
        
        return self.decoder(h)
    
    def training_step(self, batch: TensorDict, batch_idx: int) -> Tensor:
        """
        Executes a single training step using behavioral cloning loss.
        
        In PyTorch Lightning, the model defines its own training logic. This is 
        Lightning's standard pattern, in that the model knows how to train itself, 
        eliminating the need for external training loops.
        
        Args:
            batch     : TensorDict containing graph observations and expert actions
            batch_idx : Current batch index (automatically provided by Lightning)
            
        Returns:
            Scalar loss tensor for automatic backpropagation
        """
        return self._compute_loss_and_log(batch, "train")
    
    def validation_step(self, batch: TensorDict, batch_idx: int) -> Tensor:
        """
        Executes validation step for model evaluation.
        
        Lightning calls this method during validation to assess the model's
        performance on held-out data. This helps monitor generalization and
        detect overfitting during training.
        
        Args:
            batch     : TensorDict containing validation observations and actions
            batch_idx : Current batch index (automatically provided by Lightning)
            
        Returns:
            Scalar validation loss for automatic metric aggregation
        """
        return self._compute_loss_and_log(batch, "val")