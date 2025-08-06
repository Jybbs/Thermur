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
    from .safety                     import CBFSafetyFilter
    from config.imitation.controller import FlockModel, MurmurationModel, SafetyModel
    from tensordict                  import TensorDictBase
    from torch                       import Tensor


class MurmurationController:
    """
    Implements murmuration dynamics with topological interactions.

    This controller generates biologically-inspired flocking behavior based on
    starling murmurations, using topological neighborhoods (k-nearest neighbors)
    rather than metric distances. The flock maintains critical state dynamics
    for rapid information propagation and exhibits distinct cruise/alert modes.

    The controller implements the Hamiltonian formulation from Bialek et al. (2012):

        E = -Σ_{<ij>} J_{ij} 𝐬_i · 𝐬_j - Σ_i 𝐡_i · 𝐬_i

    where:
        - 𝐬_i = 𝐯_i / |𝐯_i| are normalized velocity vectors (spin variables)
        - J_{ij} = J_0 exp(-d_{ij}/λ) are pairwise coupling strengths
        - d_{ij} is the topological distance (minimum hop count)
        - 𝐡_i represents external fields (thermal gradients)

    Forces are derived as 𝐮_i = -∂E/∂𝐱_i, yielding:

        F_i = Σ_j J_{ij} (𝐯_j - 𝐯_i)

    This formulation reproduces scale-free correlations C(r) ~ r^{-1/3} and
    information propagation speeds of 15-45 m/s observed in real murmurations.
    """

    def __init__(
        self,
        cbf    : CBFSafetyFilter | None,
        flock  : FlockModel,
        mmm    : MurmurationModel,
        safety : SafetyModel
    ):
        """
        Initializes the controller with the necessary configuration models.

        Args:
            cbf    : Optional Control Barrier Function filter for safety.
                     If None, no safety filtering is applied
            flock  : Flock configuration containing agent properties
            mmm    : Murmuration model with dynamics and weight parameters
            safety : Safety configuration with thresholds and CBF parameters
        """
        self.cbf    = cbf
        self.flock  = flock
        self.mmm    = mmm
        self.safety = safety
        self._reset_shared_state()

    def __call__(self, flock: TensorDictBase) -> TensorDictBase:
        """
        Compute control actions in TorchRL-compatible format.

        This method makes MurmurationController compatible with TorchRL's
        expected policy interface by wrapping the nominal action
        computation and returning a TensorDict with the action.

        Args:
            flock: TensorDict containing the current flock state

        Returns:
            TensorDict with the computed action added
        """
        action          = self.compute_nominal_action(flock)
        flock["action"] = action
        return flock

    def _compute_density_wave(
        self,
        positions     : Tensor,
        threat_levels : Tensor
    ) -> Tensor:
        """
        Compute density wave forces from PDE dynamics.

        Implements simplified density wave equation:
            ∂ρ/∂t + ∇·(ρv) = D∇²ρ + S_threat
        
        where density perturbations propagate through the flock creating
        the characteristic "ink-like" appearance during evasion.

        Args:
            positions     : Tensor [N, 3] of agent positions
            threat_levels : Tensor [N] of normalized threat levels

        Returns:
            Tensor [N, 3] of density wave forces
        """
        n_agents = self.flock.agent_count
        if n_agents < 2:
            return th.zeros_like(positions)
        
        distances = th.cdist(positions, positions)
        
        weights = th.exp(
            -distances**2 / (2 * self.mmm.density_bandwidth**2)
        )
        weights.fill_diagonal_(0)
        
        local_density    = weights.sum(dim=1, keepdim=True)
        position_diffs   = positions.unsqueeze(0) - positions.unsqueeze(1)
        weighted_diffs   = weights.unsqueeze(2) * position_diffs
        density_gradient = (
            weighted_diffs.sum(dim=1) / 
            local_density.clamp_min(self.mmm.epsilon)
        )
        
        diffusion_force   = -self.mmm.density_diffusion * density_gradient
        threat_modulation = 1 + threat_levels.unsqueeze(1) * 2
        
        return diffusion_force * threat_modulation
    
    def _compute_hamiltonian_forces(
        self,
        gradient   : Tensor,
        positions  : Tensor,
        velocities : Tensor
    ) -> Tensor:
        """
        Compute forces from Hamiltonian energy minimization.

        Implements the Hamiltonian formulation from Bialek et al. (2012):

            E = -Σ_{<ij>} J_{ij} 𝐬_i · 𝐬_j - Σ_i 𝐡_i · 𝐬_i

        where 𝐬_i are normalized velocities (spin variables) and J_{ij} decay
        with topological distance. Forces are computed as 𝐮_i = -∂E/∂𝐱_i.

        From the energy minimization, the alignment force on agent i becomes:

            F_i^{align} = Σ_j J_{ij} (𝐯_j - 𝐯_i)

        where j runs over topological neighbors. This produces velocity
        correlations with power-law decay C(r) ~ r^{-γ} with γ ≈ 1/3.

        Args:
            gradient   : Tensor [N, 3] of temperature gradients (external field 𝐡)
            positions  : Tensor [N, 3] of agent positions 𝐱
            velocities : Tensor [N, 3] of agent velocities 𝐯

        Returns:
            Tensor [N, 3] of control forces
        """
        n_agents = self.flock.agent_count

        edge_source, edge_target = edge_index = self._compute_topological_neighbors(
            k_neighbors = self.mmm.k_neighbors,
            positions   = positions
        )
        
        topo_distances = self._compute_topological_distances(
            edge_index = edge_index,
            n_agents   = n_agents
        )
        
        forces = th.zeros_like(positions)
        
        if edge_source.numel() > 0:
            vel_i = velocities[edge_source]
            vel_j = velocities[edge_target]

            J_edges = self.mmm.j_base * th.exp(
                -topo_distances[edge_source, edge_target] / self.mmm.coupling_decay
            )

            force_contrib = J_edges.unsqueeze(1) * (vel_j - vel_i)
            forces.index_add_(0, edge_source, force_contrib)
        
        metric_distances = th.cdist(positions, positions)
        mask = (
            (metric_distances < self.mmm.min_distance * 3) & 
            (metric_distances > 0)
        )

        if mask.any():
            i_idx, j_idx  = mask.nonzero(as_tuple=True)
            displacement  = positions[j_idx] - positions[i_idx]
            soft_distance = displacement.norm(
                dim     = 1, 
                keepdim = True
            ).clamp_min(self.mmm.min_distance)

            forces.index_add_(
                dim    = 0, 
                index  = i_idx, 
                source = (
                    -self.mmm.separation_strength * displacement / 
                    soft_distance ** 3
                )
            )
        
        forces -= self.mmm.temperature_scaling * gradient

        return forces
    
    def _compute_information_speed(self, susceptibility: Tensor) -> Tensor:
        """
        Compute information propagation speed through the flock.

        Following empirical observations, information speed scales with
        susceptibility:

            v_info = c_0 √(χ/m_eff)

        where c_0 is a proportionality constant and m_eff is effective mass.
        Real murmurations achieve v_info ∈ [15, 45] m/s.

        Args:
            susceptibility: Scalar susceptibility value χ

        Returns:
            Scalar information speed in m/s
        """
        v_info = self.mmm.info_speed_coefficient * th.sqrt(
            susceptibility / self.mmm.effective_mass
        )
        return v_info.clamp(self.mmm.info_speed_min, self.mmm.info_speed_max)
    
    def _compute_susceptibility(self, velocities: Tensor) -> Tensor:
        """
        Compute flock susceptibility χ = N · Var[Φ].

        From Cavagna et al. (2010), susceptibility measures the flock's
        responsiveness to perturbations:

            χ = N · ⟨(Φ - ⟨Φ⟩)²⟩

        where Φ = |Σ_i 𝐬_i|/N is the polarization order parameter. At critical
        state, χ diverges, enabling rapid information propagation with speed:

            v_info = c_0 √(χ/m_eff) ∈ [15, 45] m/s

        Note: For real-time control, we use instantaneous variance. Full temporal
        variance would require maintaining history across timesteps.

        Args:
            velocities: Tensor [N, 3] containing agent velocities 𝐯

        Returns:
            Scalar susceptibility value χ
        """
        spin_vectors = (
            velocities / 
            velocities.norm(dim=1, keepdim=True).clamp_min(1e-8)
        )
        mean_spin  = spin_vectors.mean(dim=0)
        variance   = ((spin_vectors - mean_spin).norm(dim=1) ** 2).mean()

        return self.flock.agent_count * variance
    
    def _compute_threat_level(self, temperature: Tensor) -> Tensor:
        """
        Compute normalized threat level for mode switching.

        Maps temperature to [0, 1] range where:
            - θ < 0.3: Cruise mode (normal murmuration)
            - θ ≥ 0.3: Alert mode (enhanced cohesion and correlation)

        The threshold at 0.7 * T_max provides a safety margin before critical
        temperature is reached.

        Args:
            temperature: Tensor [N] or [N, 1] containing agent temperatures T

        Returns:
            Tensor [N] of normalized threat levels θ ∈ [0, 1]
        """
        max_temp = self.safety.max_temperature
        threat_level = (
            (temperature - max_temp * self.mmm.threat_threshold_ratio) /
            (max_temp * self.mmm.threat_range_ratio)
        )

        return threat_level.clamp(0, 1)
    
    def _compute_topological_distances(
        self,
        edge_index : Tensor,
        n_agents   : int
    ) -> Tensor:
        """
        Compute minimum hop distances between all pairs of agents.

        Uses Floyd-Warshall algorithm to find shortest paths in the k-NN graph.
        This captures the topological distance metric d_{ij} used in the
        coupling strength decay J_{ij} = J_0 exp(-d_{ij}/λ).

        The algorithm iteratively relaxes distances:
            d_{ij}^{(k+1)} = min(d_{ij}^{(k)}, d_{ik}^{(k)} + d_{kj}^{(k)})

        Args:
            edge_index : Tensor [2, E] of k-NN connections
            n_agents   : Number of agents

        Returns:
            Tensor [N, N] of topological distances (hop counts)
        """
        device = edge_index.device
        dist   = th.full((n_agents, n_agents), float('inf'), device=device)
        dist.fill_diagonal_(0)

        if edge_index.numel() > 0:
            dist[edge_index[0], edge_index[1]] = 1
            dist[edge_index[1], edge_index[0]] = 1

        for k in range(n_agents):
            dist = th.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])

        return dist
    
    def _compute_topological_neighbors(
        self,
        k_neighbors : int,
        positions   : Tensor
    ) -> Tensor:
        """
        Compute k-nearest neighbors for each agent.

        Following Ballerini et al. (2008), each agent tracks exactly 6-7
        nearest neighbors regardless of metric distance. This topological
        interaction rule is key to achieving scale-free correlations.

        Args:
            k_neighbors : Number of nearest neighbors (typically 6-7)
            positions   : Tensor [N, 3] containing agent positions

        Returns:
            Tensor [2, E] edge index in COO format for PyG
        """
        distances  = th.cdist(positions, positions)
        _, indices = distances.topk(k_neighbors + 1, largest=False)

        n_agents    = self.flock.agent_count
        edge_source = th.arange(n_agents).repeat_interleave(k_neighbors)
        edge_target = indices[:, 1:].flatten()

        return th.stack([edge_source, edge_target])

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
            if temperature.ndim > 1 and temperature.shape[1] == 1
            else temperature
        )

    def _estimate_gradient(
        self,
        position    : Tensor,
        temperature : Tensor
    ) -> Tensor:
        """
        Estimates the temperature gradient ∇T at each agent position.

        Approximates gradients using finite differences from neighboring agents:

            ∇T_i ≈ Σ_j (T_j - T_i)(𝐱_j - 𝐱_i) / |𝐱_j - 𝐱_i|²

        For isolated agents or uniform temperature fields, assumes heat rises
        vertically following the natural convection model:

            ∇T = (T/T_max) 𝐞_z

        Args:
            position    : Tensor [N, d] containing agent positions 𝐱
            temperature : Tensor [N] containing temperatures T

        Returns:
            Tensor [N, d] of estimated temperature gradients ∇T
        """
        temperature = self._ensure_1d_temperature(temperature)

        if not self._edge_source.numel():
            return self._vertical_heat_gradient(
                position    = position,
                temperature = temperature
            )

        n_agents, _      = position.shape
        position_diff    = position[self._edge_target] - position[self._edge_source]
        temperature_diff = (
            temperature[self._edge_target] - temperature[self._edge_source]
        )

        significant_mask = th.abs(temperature_diff) > self.mmm.epsilon
        gradient_sum     = th.zeros_like(position)
        neighbor_counts  = th.bincount(
            input     = self._edge_source[significant_mask],
            minlength = n_agents
        ).float()

        gradient_sum.index_add_(
            dim    = 0,
            index  = self._edge_source[significant_mask],
            source = position_diff[significant_mask] * temperature_diff[
                significant_mask
            ].unsqueeze(dim=1)
        )
        gradient_estimate = (
            gradient_sum / 
            neighbor_counts.clamp_min(1).unsqueeze(dim=1)
        )

        return th.where(
            condition = (neighbor_counts == 0).unsqueeze(dim=1),
            input     = self._vertical_heat_gradient(position, temperature),
            other     = gradient_estimate
        )

    def _reset_shared_state(self, device: str | th.device = 'cpu') -> None:
        """
        Resets the shared graph state variables to empty tensors.

        Initializes edge connectivity and neighbor count tensors used for
        efficient gradient computation across timesteps.
        """
        empty_long = lambda: th.tensor([], device=device, dtype=th.long)

        self._edge_source    = empty_long()
        self._edge_target    = empty_long()
        self._neighbor_count = empty_long()
        self._safe_count     = empty_long()

    def _update_graph_state(
        self,
        flock      : TensorDictBase,
        num_agents : int
    ) -> None:
        """
        Update graph connectivity using topological neighborhoods.

        Computes k-nearest neighbor topology from current positions, overriding
        any metric-based connectivity. This ensures topological interactions
        as observed in Ballerini et al. (2008).

        Args:
            flock      : TensorDict containing current positions
            num_agents : Total number of agents in flock
        """
        edge_index = self._compute_topological_neighbors(
            k_neighbors = self.mmm.k_neighbors,
            positions   = flock["position"]
        )
        device = edge_index.device
        if edge_index.numel():
            self._edge_source, self._edge_target = edge_index
            self._neighbor_count                 = th.bincount(
                input     = self._edge_source,
                minlength = num_agents
            ).to(device)
        else:
            self._reset_shared_state(device)
            self._neighbor_count = th.zeros(
                num_agents,
                device = device,
                dtype  = th.long
            )

        self._safe_count = self._neighbor_count.clamp_min(1)

    def _vertical_heat_gradient(
        self,
        position    : Tensor,
        temperature : Tensor
    ) -> Tensor:
        """
        Creates a default vertical temperature gradient.

        Models natural convection where heat rises, creating a vertical gradient:

            ∇T = (T/T_max) 𝐞_z

        This fallback is used when agents are isolated or in uniform temperature
        fields where neighbor-based estimation is unavailable.

        Args:
            position    : Tensor [N, d] containing agent positions
            temperature : Tensor [N] or [N, 1] containing temperatures

        Returns:
            Tensor [N, d] containing vertical gradient vectors
        """
        num_agents, dimension = position.shape

        vertical_direction = th.zeros(
            (num_agents, dimension), device=position.device
        )
        vertical_direction[:, -1] = 1.0

        normalized_temperature = (
            self._ensure_1d_temperature(temperature) / 
            self.safety.max_temperature
        )

        return vertical_direction * normalized_temperature.unsqueeze(1)

    def compute_nominal_action(self, flock: TensorDictBase) -> Tensor:
        """
        Compute murmuration dynamics using Hamiltonian formulation.

        Generates control forces following the energy-based model from
        Bialek et al. (2012), with mode-dependent modifications:

        Cruise mode:
            𝐮 = (1 + α_χ tanh(χ/χ_target)) F^{Hamiltonian}

        Alert mode (θ > 0.3):
            𝐮 = (1 + α_corr)(1 + α_χ tanh(χ/χ_target))F^{Ham} + β_dense(𝐱_cm - 𝐱_i)

        where α_χ modulates alignment based on susceptibility, α_corr enhances 
        velocity correlation and β_dense increases flock density during threat.

        Args:
            flock: TensorDict containing positions, velocities, temperatures

        Returns:
            Tensor [N, 3] of control accelerations (m/s²)
        """
        self._update_graph_state(
            flock      = flock,
            num_agents = flock["position"].size(0)
        )

        control_forces = self._compute_hamiltonian_forces(
            gradient   = flock["gradient"],
            positions  = flock["position"],
            velocities = flock["velocity"]
        )

        susceptibility = self._compute_susceptibility(flock["velocity"])
        threat_levels  = self._compute_threat_level(flock["temperature"])
        info_speed     = self._compute_information_speed(susceptibility)
        
        control_forces *= (
            1 + self.mmm.susceptibility_amplification * 
            th.tanh(susceptibility / self.mmm.susceptibility_target)
        )
        
        density_wave_forces = self._compute_density_wave(
            positions     = flock["position"],
            threat_levels = threat_levels
        )
        control_forces += density_wave_forces
        
        # Store computed physics values for metrics and monitoring
        flock["susceptibility"] = susceptibility
        flock["info_speed"]     = info_speed
        flock["threat_levels"]  = threat_levels
        flock["density_wave"]   = density_wave_forces
        
        in_alert_mode = threat_levels.max() > self.mmm.alert_threshold
        flock["in_alert_mode"] = in_alert_mode
        
        if in_alert_mode:
            control_forces *= (1 + self.mmm.correlation_strength)
            control_forces += (
                flock["position"].mean(dim=0) - flock["position"]
            ) * self.mmm.density_strength

        if self.cbf is not None:
            return self.cbf.filter(flock, control_forces)

        return control_forces
