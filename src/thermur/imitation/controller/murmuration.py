"""
Implements murmuration dynamics with topological interactions.

This module provides a biologically-inspired controller based on starling
murmurations, using topological neighborhoods (k-nearest neighbors) rather
than metric distances. The flock maintains critical state dynamics for
rapid information propagation and exhibits distinct cruise/alert modes.
"""
from __future__ import annotations
from typing     import TYPE_CHECKING

import torch as th

if TYPE_CHECKING:
    from .safety                     import SafetyFilter
    from config.imitation.controller import MurmurationModel, FlockModel, ThresholdsModel
    from tensordict                  import TensorDictBase
    from torch                       import Tensor


class MurmurationController:
    """
    Implements murmuration dynamics with topological interactions.
    
    This controller generates biologically-inspired flocking behavior based on
    starling murmurations, using topological neighborhoods (k-nearest neighbors)
    rather than metric distances. The flock maintains critical state dynamics
    for rapid information propagation and exhibits distinct cruise/alert modes.
    
    The controller implements an enhanced Hamiltonian formulation:
        E = -Σ J_ij 𝐬_i · 𝐬_j - Σ 𝐡_i · 𝐬_i
    
    where 𝐬_i are normalized velocities and J_ij decay with topological distance.
    """

    def __init__(
        self,
        flock         : FlockModel,
        mmm           : MurmurationModel,
        thresholds    : ThresholdsModel,
        safety_filter : SafetyFilter | None = None
    ):
        """
        Initializes the controller with the necessary configuration models.

        Args:
            flock         : Flock configuration containing agent properties.
            mmm           : Murmuration model with dynamics and weight parameters.
            thresholds    : Safety threshold configuration used across domains.
            safety_filter : Optional safety filter for CBF-based control limiting.
                            If None, no safety filtering is applied.
        """
        self.flock           = flock
        self.max_temperature = thresholds.max_temperature
        self.mmm             = mmm
        self.safety_filter   = safety_filter
        self.thresholds      = thresholds
        self._reset_shared_state()

    def __call__(self, flock: TensorDictBase) -> TensorDictBase:
        """
        Compute control actions in TorchRL-compatible format.

        This method makes ExpertController compatible with TorchRL's
        expected policy interface by wrapping the nominal action
        computation and returning a TensorDict with the action.

        Args:
            flock: TensorDict containing the current flock state

        Returns:
            TensorDict with the computed action added
        """
        action = self.compute_nominal_action(flock)
        flock["action"] = action
        return flock

    
    def _compute_hamiltonian_forces(
        self,
        positions  : Tensor,
        velocities : Tensor,
        gradient   : Tensor
    ) -> Tensor:
        """
        Compute forces from Hamiltonian energy minimization.
        
        Implements the Hamiltonian formulation from Bialek et al. (2012):
            E = -Σ_<ij> J_ij s_i · s_j - Σ_i h_i · s_i
        
        where s_i are normalized velocities (spin variables) and J_ij decay
        with topological distance. Forces are computed as u_i = -∂E/∂x_i.
        
        Args:
            positions  : Tensor [N, 3] of agent positions
            velocities : Tensor [N, 3] of agent velocities
            gradient   : Tensor [N, 3] of temperature gradients (external field)
            
        Returns:
            Tensor [N, 3] of control forces
        """
        n_agents = len(positions)
        device = positions.device
        
        # Get topological neighbors and distances
        edge_index = self._compute_topological_neighbors(self.mmm.k_neighbors, positions)
        edge_source, edge_target = edge_index
        
        # Compute topological distances as minimum hop count
        metric_distances = th.cdist(positions, positions)
        topo_distances = self._compute_topological_distances(edge_index, n_agents)
        
        # Compute interaction strengths J_ij
        J_0 = self.mmm.j_base
        lambda_decay = self.mmm.coupling_decay
        J_matrix = th.zeros((n_agents, n_agents), device=device)
        
        for i, j in zip(edge_source.tolist(), edge_target.tolist()):
            J_matrix[i, j] = J_0 * th.exp(-topo_distances[i, j] / lambda_decay)
        
        # Make J_matrix symmetric
        J_matrix = (J_matrix + J_matrix.T) / 2
        
        # Compute alignment forces from Hamiltonian
        forces = th.zeros_like(positions)
        
        # The key insight from Bialek et al.: birds align their velocities with neighbors
        # Force on agent i: F_i = Σ_j J_ij (v_j - v_i) for topological neighbors
        
        if edge_source.numel() > 0:
            # Get velocities for connected pairs
            vel_i = velocities[edge_source]
            vel_j = velocities[edge_target]
            
            # J_ij values with topological decay
            J_edges = J_0 * th.exp(-topo_distances[edge_source, edge_target] / lambda_decay)
            
            # Alignment force: weighted velocity difference
            force_contrib = J_edges.unsqueeze(1) * (vel_j - vel_i)
            
            # Accumulate forces
            forces.index_add_(0, edge_source, force_contrib)
        
        # Vectorized short-range repulsion
        mask = metric_distances < self.mmm.min_distance * 3
        mask.fill_diagonal_(False)
        if mask.any():
            i_idx, j_idx = th.where(mask)
            r_vec = positions[j_idx] - positions[i_idx]
            r_norm = r_vec.norm(dim=1, keepdim=True).clamp(min=self.mmm.min_distance)
            repulsion = self.mmm.w_separation * r_vec / (r_norm ** 3)
            forces.index_add_(0, i_idx, -repulsion)
        
        # Add external field term (thermal gradient)
        h_strength = self.mmm.temperature_scaling
        forces -= h_strength * gradient
        
        return forces
    
    def _compute_topological_distances(
        self,
        edge_index : Tensor,
        n_agents   : int
    ) -> Tensor:
        """
        Compute minimum hop distances between all pairs of agents.
        
        Uses Floyd-Warshall algorithm to find shortest paths in the k-NN graph.
        
        Args:
            edge_index : Tensor [2, E] of k-NN connections
            n_agents   : Number of agents
            
        Returns:
            Tensor [N, N] of topological distances (hop counts)
        """
        device = edge_index.device
        
        # Initialize distance matrix
        dist = th.full((n_agents, n_agents), float('inf'), device=device)
        dist.fill_diagonal_(0)
        
        # Set distance 1 for direct neighbors
        if edge_index.numel() > 0:
            dist[edge_index[0], edge_index[1]] = 1
            dist[edge_index[1], edge_index[0]] = 1  # Symmetric
        
        # Floyd-Warshall algorithm
        for k in range(n_agents):
            dist = th.min(dist, dist[:, k:k+1] + dist[k:k+1, :])
        
        return dist
    
    def _compute_topological_neighbors(
        self,
        k         : int,
        positions : Tensor
    ) -> Tensor:
        """
        Compute k-nearest neighbors for each agent using topological distance.
        
        Args:
            k         : Number of nearest neighbors to connect
            positions : Tensor [N, 3] containing agent positions
            
        Returns:
            Tensor [2, E] edge index in COO format for PyG
        """
        distances   = th.cdist(positions, positions)
        _, indices  = distances.topk(k + 1, largest=False)
        edge_source = []
        edge_target = []
        
        for i in range(len(positions)):
            for j in indices[i, 1:]:
                edge_source.append(i)
                edge_target.append(j.item())
                
        return th.tensor(
            data   = [edge_source, edge_target], 
            device = positions.device,
            dtype  = th.long
        )
    
    def _compute_susceptibility(self, velocities: Tensor) -> Tensor:
        """
        Compute flock susceptibility χ = N · Var[Φ].
        
        Susceptibility measures the flock's responsiveness to perturbations.
        At critical state, χ diverges, enabling rapid information propagation.
        
        Note: For real-time control, we use instantaneous variance as an
        approximation. Full temporal variance would require maintaining
        history across timesteps, which is better suited for offline analysis.
        
        Args:
            velocities : Tensor [N, 3] containing agent velocities
            
        Returns:
            Scalar susceptibility value
        """
        normalized_vels = velocities / velocities.norm(dim=1, keepdim=True).clamp(min=1e-8)
        polarization = normalized_vels.mean(dim=0)
        
        # Instantaneous variance approximation
        velocity_var = ((normalized_vels - polarization).norm(dim=1) ** 2).mean()
        
        return len(velocities) * velocity_var
    
    def _compute_threat_level(self, temperature: Tensor) -> Tensor:
        """
        Compute normalized threat level for mode switching.
        
        Maps temperature to [0, 1] range where 0 is safe and 1 is critical.
        
        Args:
            temperature : Tensor [N] or [N, 1] containing agent temperatures
            
        Returns:
            Tensor [N] of normalized threat levels
        """
        temp_normalized = (temperature - self.thresholds.max_temperature * 0.7) / (
            self.thresholds.max_temperature * 0.3
        )
        
        return temp_normalized.clamp(0, 1)

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
        if not self._edge_source.numel():
            return self._vertical_heat_gradient(
                position    = position,
                temperature = temperature
            )

        # Calculate the neighbor-based gradient for all agents
        n, _      = position.shape
        pos_diff  = position[self._edge_target]    - position[self._edge_source]
        temp_diff = temperature[self._edge_target] - temperature[self._edge_source]

        # Sum weighted positions and count significant neighbors
        sig_mask   = th.abs(temp_diff) > self.mmm.epsilon
        grad_sum   = th.zeros_like(position)
        sig_counts = th.bincount(
            input     = self._edge_source[sig_mask],
            minlength = n
        ).float()

        grad_sum.index_add_(
            dim    = 0,
            index  = self._edge_source[sig_mask],
            source = pos_diff[sig_mask] * temp_diff[sig_mask].unsqueeze(dim=1)
        )
        grad_neighbors = grad_sum / th.clamp(sig_counts, min=1).unsqueeze(dim=1)

        return th.where(
            condition = (sig_counts == 0).unsqueeze(dim=1),
            input     = self._vertical_heat_gradient(position, temperature),
            other     = grad_neighbors
        )

    def _reset_shared_state(self, device: str | th.device = 'cpu'):
        """
        Resets the shared graph state variables to empty tensors.
        """
        empty = lambda: th.tensor([], device=device, dtype=th.long)
        self._edge_source    = empty()
        self._edge_target    = empty()
        self._neighbor_count = empty()
        self._safe_count     = empty()

    def _update_graph_state(
        self,
        flock      : TensorDictBase,
        num_agents : int
    ):
        """
        Update graph connectivity using topological neighborhoods.
        
        Computes k-nearest neighbor topology from current positions.
        
        Args:
            flock      : TensorDict containing current positions
            num_agents : Total number of agents in flock
        """
        positions = flock["position"]
        edge_index = self._compute_topological_neighbors(
            k         = self.mmm.k_neighbors,
            positions = positions
        )
        
        device = edge_index.device
        if edge_index.numel():
            self._edge_source, self._edge_target = edge_index
            self._neighbor_count = th.bincount(
                self._edge_source,
                minlength = num_agents
            ).to(device)
        else:
            self._reset_shared_state(device)
            self._neighbor_count = th.zeros(
                num_agents,
                device = device,
                dtype  = th.long
            )
            
        self._safe_count = th.clamp(self._neighbor_count, min=1)

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
        vertical = th.nn.functional.one_hot(
            num_classes = dim,
            tensor      = th.full(
                size       = (num_agents,),
                fill_value = dim - 1,
                device     = position.device,
                dtype      = th.long
            )
        ).float()

        # Scale by normalized temperature
        norm_temp = self._ensure_1d_temperature(temperature) / self.max_temperature
        return vertical * norm_temp.unsqueeze(1)

    def compute_nominal_action(self, flock: TensorDictBase) -> Tensor:
        """Compute murmuration dynamics using Hamiltonian formulation."""
        self._update_graph_state(flock, flock["position"].size(0))
        
        # Use rigorous Hamiltonian formulation from Bialek et al. (2012)
        u_nominal = self._compute_hamiltonian_forces(
            positions  = flock["position"],
            velocities = flock["velocity"],
            gradient   = flock["gradient"]
        )
        
        # Mode switching in Hamiltonian: modify coupling strength
        threat_levels = self._compute_threat_level(flock["temperature"])
        in_alert_mode = threat_levels.max() > self.mmm.alert_threshold
        
        # Store alert mode state for metrics
        flock["in_alert_mode"] = in_alert_mode
        
        if in_alert_mode:
            # Strengthen interactions in alert mode by amplifying forces
            # This effectively increases J_0 or decreases λ
            u_nominal = u_nominal * (1 + self.mmm.correlation_strength)
            
            # Add density-increasing force toward center of mass
            center_of_mass = flock["position"].mean(dim=0)
            density_force = (center_of_mass - flock["position"]) * self.mmm.density_strength
            u_nominal = u_nominal + density_force
        
        if self.safety_filter is not None:
            return self.safety_filter.filter(flock, u_nominal)
        return u_nominal
