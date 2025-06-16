"""
Implements the 'expert' flocking controller based on classic Reynolds rules
and thermal-aware potential fields.

This module provides a physics-based controller that can be used to generate
an 'optimal' trajectory dataset. A neural network policy can then be trained
via imitation learning to replicate this expert behavior.
"""
import torch

from torch import Tensor


class ExpertFlockingController:
    """
    Calculates the nominal control action `𝐮_nom` using potential fields.

    This controller computes a desired velocity for each agent by summing forces
    derived from the negative gradient of several potential functions.
    Its behavior is parameterized by configuration objects.
    """

    def __init__(
        self, 
        expert_config,
        agent_config
    ):
        """
        Initializes the controller with the necessary configuration models.

        Args:
            expert_config : Contains the weights for each potential field.
            agent_config  : Contains agent-specific properties like the maximum
                            survivable temperature.
        """
        self.expert_config = expert_config
        self.agent_config  = agent_config

    def _compute_cohesion(
        self, 
        position   : Tensor, 
        edge_index : Tensor
    ) -> Tensor:
        """
        Calculates the cohesion force vector: U_coh ∝ Σ||xᵢ - xⱼ||².
        
        This implements the cohesion component of Reynolds' flocking rules,
        creating a force that pulls each agent toward the center of mass
        of its neighbors.
        
        For each agent i, we calculate:
            x̄ᵢ = (1/|N(i)|) · Σⱼ∈N(i) xⱼ
            F_coh = x̄ᵢ - xᵢ
        
        where N(i) is the set of neighbors defined by edge_index, and
        x̄ᵢ is the center of mass of these neighbors.
        
        Args:
            position   : Tensor of shape [num_agents, dim] containing positions
            edge_index : Tensor of shape [2, num_edges] defining the communication graph
        
        Returns:
            Tensor of shape [num_agents, dim] containing cohesion force vectors
        """
        num_agents     = position.size(0)
        dim            = position.size(1)
        device         = position.device
        center_of_mass = torch.zeros_like(position)
        neighbor_count = torch.zeros(
            num_agents,
            dtype  = torch.int64, 
            device = device
        )
        
        # Extract source (`s`) and target (`t`) indices from edge_index
        s, t = edge_index
        
        # Aggregate and count neighbor positions for center of mass calculation
        for d in range(dim):
            center_of_mass[:, d].scatter_add_(0, s, position[t, d])
        
        neighbor_count.scatter_add_(0, s, torch.ones_like(s))
    
        safe_count     = torch.max(neighbor_count, torch.ones_like(neighbor_count))
        center_of_mass = center_of_mass / safe_count.unsqueeze(1)
        
        # Zero out forces for agents with no neighbors
        has_neighbors = neighbor_count > 0
        return (center_of_mass - position) * has_neighbors.unsqueeze(1).float()

    def _compute_separation(
        self, 
        position   : Tensor, 
        edge_index : Tensor
    ) -> Tensor:
        """
        Calculates the separation force vector: U_sep ∝ Σ 1/||xᵢ - xⱼ||.
        
        This implements the separation component of Reynolds' flocking rules,
        creating a repulsive force that prevents collisions between agents.
        The force magnitude is inversely proportional to distance.
        
        For each agent i and its neighbor j, we calculate:
            F_sepᵢⱼ = (xᵢ - xⱼ) / ||xᵢ - xⱼ||²
            F_sep = Σⱼ∈N(i) F_sepᵢⱼ
        
        Args:
            position   : Tensor of shape [num_agents, dim] containing positions
            edge_index : Tensor of shape [2, num_edges] defining the communication graph
        
        Returns:
            Tensor of shape [num_agents, dim] containing separation force vectors
        """
        dim        = position.size(1)
        separation = torch.zeros_like(position)
        
        # Extract source (`s`) and target (`t`) indices from edge_index
        s, t = edge_index
        
        # Calculate displacement vectors (xᵢ - xⱼ) and squared distances
        epsilon      = 1e-8
        displacement = position[s] - position[t]
        squared_dist = torch.sum(displacement * displacement, dim=1) + epsilon
        
        # Scale displacement by repulsion magnitude (1/r²)
        repulsion_mag  = 1.0 / squared_dist
        repulsion_vec  = displacement * repulsion_mag.unsqueeze(1)
        
        # Aggregate repulsion vectors for each agent
        for d in range(dim):
            separation[:, d].scatter_add_(0, s, repulsion_vec[:, d])
        
        return separation

    def _compute_alignment(
        self, 
        velocity   : Tensor, 
        edge_index : Tensor
    ) -> Tensor:
        """
        Calculates the alignment force vector: U_align ∝ Σ||vᵢ - vⱼ||².
        
        This implements the alignment component of Reynolds' flocking rules,
        creating a force that causes agents to match velocities with their
        neighbors.
        
        For each agent i, we calculate:
            v̄ᵢ = (1/|N(i)|) · Σⱼ∈N(i) vⱼ
            F_align = v̄ᵢ - vᵢ
        
        where v̄ᵢ is the average velocity of the neighbors of agent i.
        
        Args:
            velocity   : Tensor of shape [num_agents, dim] containing velocities
            edge_index : Tensor of shape [2, num_edges] defining the communication graph
        
        Returns:
            Tensor of shape [num_agents, dim] containing alignment force vectors
        """
        dim            = velocity.size(1)
        device         = velocity.device
        num_agents     = velocity.size(0)
        avg_velocity   = torch.zeros_like(velocity)
        neighbor_count = torch.zeros(
            num_agents,
            dtype  = torch.int64, 
            device = device
        )
        
        # Extract source (`s`) and target (`t`) indices from edge_index
        s, t = edge_index
        
        # Aggregate neighbor velocities and count neighbors
        for d in range(dim):
            avg_velocity[:, d].scatter_add_(0, s, velocity[t, d])
        
        neighbor_count.scatter_add_(0, s, torch.ones_like(s))
        
        safe_count   = torch.max(neighbor_count, torch.ones_like(neighbor_count))
        avg_velocity = avg_velocity / safe_count.unsqueeze(1)
        
        # Zero out forces for agents with no neighbors
        has_neighbors = neighbor_count > 0
        return (avg_velocity - velocity) * has_neighbors.unsqueeze(1).float()

    def _compute_thermal(
        self, 
        position    : Tensor, 
        temperature : Tensor
    ) -> Tensor:
        """
        Calculates the thermal repulsion force: U_therm ∝ 1/(T_max - Tᵢ).
        
        This implements a thermal-aware repulsion that prevents agents from
        entering high-temperature regions. The force magnitude increases
        sharply as the agent's temperature approaches the maximum survivable
        temperature.
        
        For each agent i, we calculate:
            F_thermᵢ = -∇T_i · 1/(T_max - T_i)
        
        where ∇T_i is the temperature gradient at the agent's position, and
        T_max is the maximum survivable temperature from agent_config.
        
        Args:
            position    : Tensor of shape [num_agents, dim] containing positions
            temperature : Tensor of shape [num_agents] containing temperature at each agent's position
        
        Returns:
            Tensor of shape [num_agents, dim] containing thermal repulsion force vectors
        """
        dim        = position.size(1)
        device     = position.device
        num_agents = position.size(0)
        max_temp   = self.agent_config.max_temperature
        
        # Calculate repulsion magnitude (increases as T approaches T_max)
        epsilon   = 1e-8
        t_margin  = torch.clamp(max_temp - temperature, min=epsilon)
        magnitude = 1.0 / t_margin
        
        # Initialize thermal force with an approximate temperature gradient
        # In a real implementation, we would compute actual gradient from environment data
        thermal_force = torch.zeros_like(position)
        
        # For demonstration, create a simple gradient approximation
        # Agents with higher temperatures create steeper gradients
        normalized_temp = temperature / max_temp
        
        # Scale the repulsion based on temperature
        for i in range(num_agents):
            # Direction: use unit vector in upward direction (assuming heat rises)
            if dim >= 3:
                direction = torch.tensor([0.0, 0.0, 1.0], device=device)
                
            else:
                direction = torch.tensor([0.0, 1.0], device=device)
                
            # Scale by normalized temperature and margin-based magnitude
            thermal_force[i] = direction * normalized_temp[i] * magnitude[i]
        
        return thermal_force

    def compute_nominal_action(self, sd) -> Tensor:
        """
        Computes the collective nominal control action for the entire swarm.

        This method orchestrates the calculation of all potential fields and
        combines them in a weighted sum to produce the final velocity command.

        Args:
            sd: The swarm data containing the swarm's current state including
                position, velocity, temperature, and edge_index tensors.

        Returns:
            A tensor of nominal velocity commands `𝐮_nom` for all agents.
        """
        u_coh   = self._compute_cohesion(sd.position, sd.edge_index)
        u_sep   = self._compute_separation(sd.position, sd.edge_index)
        u_align = self._compute_alignment(sd.velocity, sd.edge_index)
        u_therm = self._compute_thermal(sd.position, sd.temperature)

        return (
            self.expert_config.w_cohesion     * u_coh
            + self.expert_config.w_separation * u_sep
            + self.expert_config.w_alignment  * u_align
            + self.expert_config.w_thermal    * u_therm
        )
