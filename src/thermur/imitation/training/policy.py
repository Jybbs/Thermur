"""
Defines the Graph Neural Network (GNN) policy module, π_θ.

This file contains the implementation of the `th.nn.Module` that serves as
the agent's brain. The policy, denoted π_θ, is a GNN designed to process the
flock's state, which is naturally represented as a dynamic graph. It learns to
output a nominal velocity command, 𝐮_nom, for each agent.

The architecture is explicitly designed to be configurable and to consume
`torch_geometric.data.Data` objects, which are generated from the environment's
`TensorDict` observations.
"""
from __future__            import annotations
from pytorch_lightning     import LightningModule
from torch.nn              import GRUCell, Linear, ModuleList
from torch.nn.functional   import mse_loss
from torch_geometric.data  import Batch, Data
from torch_geometric.nn    import GCNConv
from typing                import Callable, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .metrics                          import MetricsCollector
    from config.imitation.training         import ArchitectureModel
    from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig, STEP_OUTPUT
    from tensordict                        import TensorDictBase
    from torch                             import Tensor
    from torch.nn                          import Module
    from torch.optim                       import Optimizer
    from torch.optim.lr_scheduler          import LRScheduler

import torch as th


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
        collector        : MetricsCollector,
        optimizer        : Callable[..., Optimizer],
        scheduler        : Callable[..., LRScheduler],
        scheduler_metric : str,
        training_metric  : str
    ):
        """
        Initializes the GNN policy network.

        Args:
            architecture     : Configuration for GNN architecture including hidden
                               dimensions, number of layers, activation function, and
                               I/O dimensions.
            collector        : Centralized metrics collection and management system.
            optimizer        : Pre-configured optimizer partial from hydra-zen.
            scheduler        : Pre-configured scheduler partial from hydra-zen.
            scheduler_metric : Metric name for learning rate scheduler to monitor.
            training_metric  : Metric name to monitor for training loss.
        """
        super().__init__()
        self.architecture     = architecture
        self.collector        = collector
        self.optimizer        = optimizer
        self.scheduler        = scheduler
        self.scheduler_metric = scheduler_metric
        self.training_metric  = training_metric

        self.save_hyperparameters(ignore=["collector"])
        self.activation = getattr(th.nn, architecture.activation)()
        self.convs      = self._build_module_list(architecture, GCNConv)
        self.grus       = self._build_module_list(architecture, GRUCell)
        self.decoder    = Linear(architecture.hidden_dim, 3)
        self.encoder    = Linear(13, architecture.hidden_dim)
        
        self._edge_offset_cache      = {}
        self._batch_assignment_cache = {}

    @th.jit.ignore
    def _batch_to_data(self, batch: TensorDictBase) -> Batch:
        """
        Convert TensorDict batch to PyTorch Geometric graph format.

        Transforms agent-based observations into graph representations suitable for
        GNN processing. Creates a disjoint union of graphs with proper node indexing
        for batch processing.

        The node feature vector 𝐱ᵢ ∈ ℝ¹³ for agent i consists of:
            𝐱ᵢ = [𝐩ᵢ; 𝐯ᵢ; θᵢ; ∇θᵢ; 𝐰ᵢ]
        
        where:
            - 𝐩ᵢ ∈ ℝ³  : Position vector
            - 𝐯ᵢ ∈ ℝ³  : Velocity vector  
            - θᵢ ∈ ℝ   : Temperature scalar
            - ∇θᵢ ∈ ℝ³ : Temperature gradient
            - 𝐰ᵢ ∈ ℝ³  : Wind velocity

        Args:
            batch: TensorDict containing flock observations with shapes [B, N, d]
                   where B is batch size, N is number of agents, d is feature dimension

        Returns:
            PyG Batch containing all graphs with concatenated node features and edges
        """
        features       = ["position", "velocity", "temperature", "gradient", "wind"]
        batch_size     = batch["position"].shape[0]
        num_agents     = batch["position"].shape[1]
        agent_features = th.cat([batch[f] for f in features], dim=-1)
        x, _           = self._flatten_agent_batch(agent_features)
        cache_key      = (batch_size, num_agents, x.device.type)
        
        offsets = self._edge_offset_cache.setdefault(
            cache_key,
            th.arange(batch_size, device=x.device).unsqueeze(1) * num_agents
        )
        batch_assignment = self._batch_assignment_cache.setdefault(
            cache_key,
            th.arange(batch_size, device=x.device).repeat_interleave(num_agents)
        )
        edge_indices = batch["edge_index"]
        
        if (edge_indices.numel() > 0) and (edge_indices.shape[-1] > 0):
            adjusted_edges = edge_indices + offsets.unsqueeze(1)
            edge_index     = adjusted_edges.transpose(0, 1).reshape(2, -1)
        else:
            edge_index     = th.empty((2, 0), dtype=th.long, device=x.device)
        
        return Batch(
            batch      = batch_assignment,
            edge_index = edge_index, 
            x          = x, 
        )

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
        batch       : TensorDictBase,
        is_training : bool
    ) -> Tensor:
        """
        Compute behavioral cloning loss and update metrics.

        Implements the imitation learning objective:
         
            ℒ(θ) = 𝔼[(π_θ(s) - a*)²]
        where π_θ represents the learned policy and a* the expert demonstrations.
        
        The loss computation operates over all agents in the batch, with targets
        reshaped from [B, N, 3] to [B*N, 3] to match PyTorch Geometric's 
        concatenated node format.

        Args:
            batch       : TensorDict containing graph observations and expert actions
            is_training : Whether this is training (True) or validation (False)

        Returns:
            Scalar MSE loss for gradient computation
        """
        data        = self._batch_to_data(batch)
        predictions = self(data)
        targets     = batch["action"].reshape(-1, 3)
        loss        = mse_loss(predictions, targets)

        self.collector.update_imitation_metrics(
            batch       = batch,
            is_training = is_training,
            predictions = predictions,
            targets     = targets
        )

        self.collector.log_all_metrics(
            is_training = is_training,
            module      = self,
            step_data   = {
                "loss"        : loss,
                "predictions" : predictions,
                "targets"     : targets
            }
        )

        return loss
    
    def _flatten_agent_batch(self, tensor: Tensor) -> tuple[Tensor, int]:
        """
        Flatten hierarchical agent batches for node-level processing.
        
        Transforms multi-agent batch tensors from [batch, agents, features]
        format used by replay buffers to [batch*agents, features] format
        required by graph neural networks.
        
        Args:
            tensor: Input with shape [batch, agents, features] for batched
                    trajectories, [agents, features] for single timesteps,
                    or [features] for single agents
        
        Returns:
            Tuple of (flattened_tensor, n_samples) where flattened has shape
            [total_samples, features] and n_samples counts individual agents
        """
        if tensor.dim() == 3:
            shape = tensor.shape
            return tensor.reshape(-1, shape[2]), shape[0] * shape[1]
        
        elif tensor.dim() == 2:
            return tensor, tensor.shape[0]
        
        else:
            return tensor.unsqueeze(0), 1

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """
        Configures the optimizer and learning rate scheduler for training.

        Lightning calls this method to set up optimizers and learning rate
        schedulers. The optimizer and scheduler are pre-configured partials
        from hydra-zen that just need their final parameters.

        Returns:
            Configuration for optimizer and learning rate scheduler
        """
        optimizer: Optimizer   = self.optimizer(params=self.parameters())
        scheduler: LRScheduler = self.scheduler(optimizer=optimizer)

        config: OptimizerLRSchedulerConfig = {
            "optimizer"    : optimizer,
            "lr_scheduler" : {
                "scheduler" : scheduler,
                "monitor"   : self.scheduler_metric
            }
        }
        return config

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
        
        if x.dim() == 2 and x.device.type == 'mps':
            x = x.contiguous(th.channels_last) if x.shape[-1] % 4 == 0 else x
        
        h = self.activation(self.encoder(x))

        for conv, gru in zip(self.convs, self.grus, strict=True):
            conv_out = conv(h, edge_index)
            h        = gru(self.activation(conv_out), h)

        return self.decoder(h)
    
    def on_fit_start(self):
        """
        Lightning lifecycle hook called at the beginning of training.
        
        Ensures all metric collections are on the same device as the model
        to prevent device mismatch errors during metric computation. This is
        necessary because TorchMetrics creates metrics on CPU by default,
        but Lightning may move the model to GPU/MPS.
        """
        for metric_name in [
            'train_imitation',  'val_imitation', 
            'train_evaluation', 'val_evaluation'
        ]:
            metrics = getattr(self.collector, metric_name, None)
            if metrics is None:
                raise AttributeError(
                    f"MetricsCollector missing required '{metric_name}' metrics"
                )
            setattr(self.collector, metric_name, metrics.to(self.device))

    def training_step(self, batch: TensorDictBase, batch_idx: int) -> STEP_OUTPUT:
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
        return self._compute_loss_and_log(batch, True)

    def validation_step(self, batch: TensorDictBase, batch_idx: int) -> STEP_OUTPUT:
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
        return self._compute_loss_and_log(batch, False)
