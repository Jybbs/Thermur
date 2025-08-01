"""
Implements the expert controller based on classic Reynolds rules
and thermal-aware potential fields.

This module provides a physics-based expert controller that generates
optimal trajectory datasets for imitation learning. The expert combines
Reynolds flocking rules with thermal constraints to demonstrate safe
collective behavior.
"""
from .safety                     import SafetyFilter
from config.imitation.controller import ExpertModel, FlockModel, ThresholdsModel
from tensordict                  import TensorDict
from torch                       import Tensor

import torch
import torch.nn.functional as F


class ExpertController:
    """
    Calculates the nominal control action `𝐮_nom` using potential fields.

    This controller computes a desired velocity for each agent by summing forces
    derived from the negative gradient of several potential functions, where the 
    individual potential components follow classical Reynolds rules:
        - U_coh^(i)     = (1/2) · Σⱼ∈N(i) ||𝐱ᵢ - 𝐱ⱼ||²
        - U_sep^(i)     = Σⱼ∈N(i) 1/||𝐱ᵢ - 𝐱ⱼ||
        - U_align^(i,j) = (1/2) · ||𝐯ᵢ - 𝐯ⱼ||²
        - U_therm^(i)   = 1/(T_max - T_i)
    
    The nominal control action is then 𝐮_nom^(i) = -∇ₓᵢU(𝐒ₜ)
    """

    def __init__(
        self,
        expert        : ExpertModel,
        flock         : FlockModel,
        thresholds    : ThresholdsModel,
        safety_filter : SafetyFilter | None = None
    ):
        """
        Initializes the controller with the necessary configuration models.

        Args:
            expert        : Contains both Reynolds weights and numerical parameters
                            for stable force calculations.
            flock         : Flock configuration containing agent properties.
            safety_filter : Optional safety filter for CBF-based control limiting.
                            If None, no safety filtering is applied.
            thresholds    : Safety threshold configuration used across domains.
        """
        self.expert          = expert
        self.flock           = flock
        self.max_temperature = thresholds.max_temperature
        self.safety_filter   = safety_filter
        self.thresholds      = thresholds
        self._reset_shared_state()

    def _compute_alignment(self, velocity: Tensor) -> Tensor:
        """
        Calculates the alignment force vector for each agent.
        
        The alignment term implements the third of Reynolds' flocking rules,
        creating a force that causes agents to match velocities with their
        neighbors. For each agent i, the average velocity of its neighborhood is:
        
            𝐯̄ᵢ = (1/|N(i)|) · Σⱼ∈N(i) 𝐯ⱼ
            
        The resulting alignment force is the difference between this average
        and the agent's current velocity:
            
            𝐅_align = 𝐯̄ᵢ - 𝐯ᵢ
            
        This creates a tendency for agents to synchronize their motion,
        leading to the coherent movement patterns observed in natural flocks.
        
        Args:
            velocity: Tensor [N, dim] containing agent velocities 𝐯
        
        Returns:
            Tensor [N, dim] of alignment force vectors for all agents
        """
        avg_velocity = torch.zeros_like(velocity)
        avg_velocity.index_add_(
            dim    = 0, 
            index  = self._edge_source, 
            source = velocity[self._edge_target]
        )
        avg_velocity = torch.divide(avg_velocity, self._safe_count.unsqueeze(1))

        # Apply force only to agents with neighbors
        has_neighbors = (self._neighbor_count > 0).float().unsqueeze(1)
        return (avg_velocity - velocity) * has_neighbors

    def _compute_cohesion(self, position: Tensor) -> Tensor:
        """
        Calculates the cohesion force vector for each agent.
        
        The cohesion term implements the first of Reynolds' flocking rules,
        creating an attractive force toward the center of mass of an agent's
        neighbors. For each agent i, the center of mass of its neighborhood is:
        
            𝐱̄ᵢ = (1/|N(i)|) · Σⱼ∈N(i) 𝐱ⱼ
            
        The resulting cohesion force is the vector pointing from the agent's
        position toward this center of mass:
            
            𝐅_coh = 𝐱̄ᵢ - 𝐱ᵢ
            
        The force magnitude increases with distance from the center of mass,
        creating a tendency for the flock to maintain cohesion.
        
        Args:
            position: Tensor [N, dim] containing agent positions 𝐱
        
        Returns:
            Tensor [N, dim] of cohesion force vectors for all agents
        """
        center_of_mass = torch.zeros_like(position)
        center_of_mass.index_add_(
            dim    = 0, 
            index  = self._edge_source, 
            source = position[self._edge_target]
        )

        center_of_mass = torch.divide(
            input = center_of_mass, 
            other = self._safe_count.unsqueeze(1)
        )

        # Apply force only to agents with neighbors
        has_neighbors = (self._neighbor_count > 0).float().unsqueeze(1)
        return (center_of_mass - position) * has_neighbors

    def _compute_separation(self, position: Tensor) -> Tensor:
        """
        Calculates the separation force vector for each agent.
        
        The separation term implements the second of Reynolds' flocking rules,
        creating a repulsive force that prevents collisions between agents.
        For each agent i and its neighbor j, we calculate a repulsion vector:
        
            𝐅_sepᵢⱼ = (𝐱ᵢ - 𝐱ⱼ) / ||𝐱ᵢ - 𝐱ⱼ||²
            
        The total separation force is the sum of these repulsions:
        
            𝐅_sep = Σⱼ∈N(i) 𝐅_sepᵢⱼ
            
        The force magnitude is inversely proportional to the squared distance,
        creating a stronger repulsion between agents that are close to each other.
        
        Args:
            position: Tensor [N, dim] containing agent positions 𝐱
        
        Returns:
            Tensor [N, dim] of separation force vectors for all agents
        """
        # Calculate displacement vectors and distances
        rel_pos  = position[self._edge_source] - position[self._edge_target]
        distance = torch.norm(
            dim     = 1, 
            input   = rel_pos, 
            keepdim = True
        )

        # Apply minimum distance and calculate repulsion
        distance  = torch.clamp(distance, self.expert.min_distance)
        repulsion = torch.divide(
            input = rel_pos, 
            other = distance.pow(2) + self.expert.epsilon
        )

        # Sum repulsion vectors for each agent
        separation = torch.zeros_like(position)
        separation.index_add_(
            dim    = 0, 
            index  = self._edge_source, 
            source = repulsion
        )

        return separation
    
    def _compute_thermal(
        self,
        position    : Tensor,
        temperature : Tensor,
        gradient    : Tensor | None = None
    ) -> Tensor:
        """
        Calculates the thermal repulsion force for each agent.
        
        This implements a thermal-aware repulsion that prevents agents from
        entering high-temperature regions. The force magnitude increases
        sharply as the agent's temperature approaches T_max, creating a
        strong barrier against thermal damage.
        
        For each agent i, the thermal repulsion is calculated as:
        
            𝐅_thermᵢ = -∇T_i · scale / (T_max - T_i)
            
        where ∇T_i is the temperature gradient at the agent's position,
        T_max is the maximum survivable temperature, and scale is a
        configurable parameter controlling the overall repulsion strength.
        
        Args:
            position    : Tensor [N, dim] containing agent positions 𝐱
            temperature : Tensor [N] or [N, 1] containing temperatures T
            gradient    : Optional tensor [N, dim] of pre-computed temperature
                          gradients ∇T. If None, gradients are estimated.
        
        Returns:
            Tensor [N, dim] of thermal repulsion force vectors for all agents
        """
        temperature = self._ensure_1d_temperature(temperature)
        t_margin    = torch.clamp(
            input = self.max_temperature - temperature,
            min   = self.expert.epsilon
        )

        magnitude = torch.divide(
            input = self.expert.temperature_scaling, 
            other = t_margin
        )

        gradient = (
            gradient if gradient is not None 
            else self._estimate_gradient(
                position    = position, 
                temperature = temperature
            )
        )

        # Force points away from high temperatures
        return -gradient * magnitude.unsqueeze(1)

    def _ensure_1d_temperature(self, temperature: Tensor) -> Tensor:
        """
        Ensures temperature tensor is 1D by squeezing if it's [N, 1].
        
        Args:
            temperature: Tensor [N] or [N, 1] containing temperatures
            
        Returns:
            Tensor [N] with any singleton dimensions removed
        """
        return (
            temperature.squeeze(1) 
            if temperature.dim() > 1 and temperature.size(1) == 1 
            else temperature
        )

    def _estimate_gradient(
        self,
        position    : Tensor,
        temperature : Tensor
    ) -> Tensor:
        """
        Estimates the temperature gradient ∇T at each agent position.
        
        This method uses a vectorized approach to approximate gradients using
        neighboring agent data. The gradient at each point represents the
        direction of steepest temperature increase.
        
        For agents with neighbors, the gradient is estimated by calculating
        finite differences in position weighted by temperature differentials.
        For isolated agents or those in uniform temperature fields, a fallback
        model assumes heat rises vertically, scaled by normalized temperature.
        
        Args:
            position    : Tensor [N, dim] containing agent positions 𝐱
            temperature : Tensor [N] containing temperatures T
            
        Returns:
            Tensor [N, dim] of estimated temperature gradients ∇T
        """
        temperature = self._ensure_1d_temperature(temperature)

        # Handle the edge case of a completely disconnected graph
        if self._edge_source is None or not self._edge_source.numel():
            return self._vertical_heat_gradient(
                position    = position, 
                temperature = temperature
            )

        # Calculate the neighbor-based gradient for all agents
        num_agents, _ = position.shape
        pos_diff      = position[self._edge_target]    - position[self._edge_source]
        temp_diff     = temperature[self._edge_target] - temperature[self._edge_source]
        
        # Sum weighted positions and count significant neighbors
        sig_mask   = torch.abs(temp_diff) > self.expert.epsilon
        grad_sum   = torch.zeros_like(position)
        sig_counts = torch.bincount(
            input     = self._edge_source[sig_mask],
            minlength = num_agents
        ).float()

        grad_sum.index_add_(
            dim    = 0,
            index  = self._edge_source[sig_mask],
            source = pos_diff[sig_mask] * temp_diff[sig_mask].unsqueeze(dim=1)
        )
        grad_neighbors = torch.divide(
            input = grad_sum,
            other = torch.clamp(sig_counts, min=1).unsqueeze(dim=1)
        )

        return torch.where(
            condition = (sig_counts == 0).unsqueeze(dim=1), 
            input     = self._vertical_heat_gradient(position, temperature), 
            other     = grad_neighbors
        )

    def _reset_shared_state(self):
        """
        Resets the shared graph state variables to None.
        """
        self._edge_source    = None
        self._edge_target    = None
        self._neighbor_count = None
        self._safe_count     = None

    def _update_graph_state(
        self, 
        edge_index : Tensor, 
        num_agents : int
    ):
        """
        Updates shared state for graph calculations across Reynolds rules.
        
        Args:
            edge_index : Tensor defining the communication graph topology Gₜ = (V, Eₜ)
            num_agents : The total number of agents N in the flock
        """
        if edge_index.numel():
            self._edge_source, self._edge_target = edge_index
        else:
            # Handle empty graph case to prevent errors
            device            = edge_index.device
            self._edge_source = torch.tensor([], dtype=torch.long, device=device)
            self._edge_target = torch.tensor([], dtype=torch.long, device=device)

        self._neighbor_count = torch.bincount(
            self._edge_source,
            minlength = num_agents
        ).to(edge_index.device)

        self._safe_count = torch.clamp(self._neighbor_count, min=1)

    def _vertical_heat_gradient(
        self,
        position    : Tensor,
        temperature : Tensor
    ) -> Tensor:
        """
        Creates a default vertical temperature gradient.
        
        When neighborhood-based gradient estimation is unavailable, this creates
        a gradient that points upward (assuming heat rises), scaled by the
        normalized temperature of each agent.
        
        Args:
            position    : Tensor [N, dim] containing agent positions
            temperature : Tensor [N] or [N, 1] containing temperatures
            
        Returns:
            Tensor [N, dim] containing vertical gradient vectors
        """
        num_agents, dim = position.shape

        # Create unit vectors pointing up in the last dimension
        vertical = F.one_hot(
            num_classes = dim,
            tensor      = torch.full(
                size       = (num_agents,),
                fill_value = dim - 1,
                device     = position.device,
                dtype      = torch.long
            )
        ).float()

        # Scale by normalized temperature
        norm_temp = torch.divide(
            input = self._ensure_1d_temperature(temperature),
            other = self.max_temperature
        )
        return vertical * norm_temp.unsqueeze(1)
    
    def compute_nominal_action(self, flock: TensorDict) -> Tensor:
        """
        Computes the collective nominal control action for the entire flock.

        This method calculates the weighted sum of forces from all potential
        fields to produce the final velocity command 𝐮_nom. The weights come
        from the `control` config, balancing the influence of each behavioral 
        component.
        
        If a safety_filter is provided, the nominal control action is passed
        through a Control Barrier Function to ensure thermal safety constraints
        are satisfied, resulting in a safety-certified action 𝐮*.

        Args:
            flock: The flock data containing the flock's current state including
                   position, velocity, temperature, and edge_index tensors.

        Returns:
            A tensor of velocity commands for all agents.
        """
        self._reset_shared_state()
        self._update_graph_state(flock["edge_index"], flock["position"].size(0))

        # Compute the nominal control based on Reynolds rules and thermal potential
        forces = [
            (self.expert.w_cohesion,   self._compute_cohesion(flock["position"])),
            (self.expert.w_separation, self._compute_separation(flock["position"])),
            (self.expert.w_alignment,  self._compute_alignment(flock["velocity"])),
            (self.expert.w_thermal,    self._compute_thermal(
                gradient    = flock.get("gradient", None),
                position    = flock["position"],
                temperature = flock["temperature"]
            ))
        ]
        
        u_nominal = sum(weight * force for weight, force in forces)
        
        return (
            self.safety_filter.filter(flock, u_nominal)
            if self.safety_filter is not None
            else u_nominal
        )
