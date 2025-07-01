"""
Defines the Graph Neural Network (GNN) policy module, π_θ.

This file contains the implementation of the `torch.nn.Module` that serves as
the agent's brain. The policy, denoted π_θ, is a GNN designed to process the
swarm's state, which is naturally represented as a dynamic graph. It learns to
output a nominal velocity command, 𝐮_nom, for each agent.

The architecture is explicitly designed to be configurable and to consume
`torch_geometric.data.Data` objects, which are generated from the environment's
`TensorDict` observations.
"""
from torch                     import Tensor
from torch.nn                  import GRUCell, Linear, Module, ModuleList, ReLU, SiLU, Tanh
from torch_geometric.data      import Data
from torch_geometric.nn        import GCNConv
from configs.imitation.schemas import LearningModel


class GNNPolicy(Module):
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
        learning_config : LearningModel
    ):
        """
        Initializes the GNN policy network.

        Args:
            learning_config: A learning configuration instance containing
                             architectural hyperparameters like hidden dimension,
                             number of layers, activation function, and I/O dimensions.
        """
        super().__init__()
        self.learning_config = learning_config
        
        # Extract dimensions from config
        in_dim  = learning_config.input_dim
        out_dim = learning_config.output_dim

        # Maps raw node features [𝐩, 𝐯, T, ∇T, E] to the hidden dimension.
        self.encoder = Linear(in_dim, learning_config.hidden_dim)

        # A stack of GNN layers and recurrent cells for state updates.
        self.convs = ModuleList()
        self.grus  = ModuleList()
        for _ in range(learning_config.num_layers):
            self.convs.append(GCNConv(learning_config.hidden_dim, learning_config.hidden_dim))
            self.grus.append(GRUCell(learning_config.hidden_dim, learning_config.hidden_dim))

        # Maps the final hidden state to a nominal action vector 𝐮_nom.
        self.decoder = Linear(learning_config.hidden_dim, out_dim)

        # --- Activation Function ---
        self.activation = {
            "relu" : ReLU, 
            "silu" : SiLU, 
            "tanh" : Tanh
        }[learning_config.activation]()

    def forward(self, data: Data) -> Tensor:
        """
        Performs the forward pass through the GNN.

        Args:
            data: A `torch_geometric.data.Data` object containing the batched
                  graph state of the swarm, with `x` (node features) and
                  `edge_index` (connectivity).

        Returns:
            A tensor of shape (num_nodes, out_dim) representing the nominal
            velocity command for each agent in the batch.
        """
        x, edge_index = data.x, data.edge_index

        # Encode initial node features into a latent representation.
        h = self.activation(self.encoder(x))

        # Iteratively process through message-passing layers.
        for conv, gru in zip(self.convs, self.grus):
            # Aggregate information from neighbors via GNN convolution.
            message = self.activation(conv(h, edge_index))
            # Update the node's hidden state using the aggregated message.
            h = gru(message, h)

        # Decode the final hidden state to produce the control action.
        action = self.decoder(h)

        return action
