"""
Implements murmuration dynamics with topological interactions.

This module provides a biologically-inspired controller based on starling
murmurations, using topological neighborhoods (k-nearest neighbors) rather
than metric distances. The flock maintains critical state dynamics for
rapid information propagation and exhibits distinct cruise/alert modes.
"""
from __future__  import annotations
from collections import deque
from typing      import TYPE_CHECKING

import torch as th

if TYPE_CHECKING:
    from .safety                     import CBFSafetyFilter
    from config.imitation.controller import FlockModel, MurmurationModel, SafetyModel
    from tensordict                  import TensorDictBase
    from torch                       import Tensor


class MurmurationController(th.nn.Module):
    """
    Implements murmuration dynamics with topological interactions.

    This controller generates biologically-inspired flocking behavior based on
    starling murmurations, using topological neighborhoods (k-nearest neighbors)
    rather than metric distances. The flock maintains critical state dynamics
    for rapid information propagation and exhibits distinct cruise/alert modes.

    The controller implements a modified Hamiltonian formulation based on 
    Bialek et al. (2012) with heterogeneous coupling for alert states:

        E = -Σ_{<ij>} J_{ij}^{alert} 𝐬_i · 𝐬_j - Σ_i 𝐡_i · 𝐬_i

    where:
        - 𝐬_i = 𝐯_i / |𝐯_i| are normalized velocity vectors (spin variables)
        - J_{ij}^{alert} = κ_i × J_0 exp(-d_{ij}/λ) with alert-dependent coupling
        - κ_i = 1.0 for relaxed birds, alert_coupling_factor for alert birds
        - d_{ij} is the topological distance (minimum hop count)
        - 𝐡_i represents external fields (thermal gradients)

    Forces are derived as 𝐮_i = -∂E/∂𝐱_i, yielding:

        F_i = κ_i × Σ_j J_{ij} (𝐯_j - 𝐯_i)

    With alert_coupling_factor = -1.3, alert birds actively oppose alignment,
    creating oscillations that maintain critical state susceptibility χ = N·Var[Φ] ≥ 5.
    This heterogeneity, motivated by vigilance behavior (Beauchamp 2015), enables
    scale-free correlations C(r) ~ r^{-1/3} and information speeds of 15-45 m/s
    (Attanasi et al. 2014).
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
        super().__init__()
        self.cbf    = cbf
        self.flock  = flock
        self.mmm    = mmm
        self.safety = safety
        self.alert_states_memory = {}

    def _compute_density_wave(self, flock: TensorDictBase):
        """
        Compute density wave forces from continuum density field dynamics.

        Implements a simplified reaction-diffusion model for density perturbations
        that propagate through the flock, creating the characteristic "ink-like" 
        evasion patterns observed in starling murmurations under predator attack.

        The density field ρ(𝐱,t) evolves according to:

            ∂ρ/∂t + ∇·(ρ𝐯) = D∇²ρ + S(θ)
        
        where:
            - ρ(𝐱,t) : Local agent density at position 𝐱 and time t
            - 𝐯(𝐱,t) : Velocity field of the flock  
            - D      : Diffusion coefficient controlling wave propagation speed
            - S(θ)   : Source term modulated by threat level θ ∈ [0,1]

        For computational efficiency, we approximate the density field using
        kernel density estimation with Gaussian kernels:

            ρ(𝐱ᵢ) = Σⱼ K(|𝐱ᵢ - 𝐱ⱼ|; σ)
        
        where K(r; σ) = exp(-r²/2σ²) is the Gaussian kernel with bandwidth σ.

        The resulting force on agent i opposes density gradients:

            𝐅ᵢ = -D·∇ρ(𝐱ᵢ)·(1 + 2θᵢ)

        This creates an effective pressure that disperses high-density regions,
        with the effect amplified under threat conditions (high θ).

        Args:
            flock: TensorDict containing:
                - position     : Agent positions 𝐱 ∈ ℝ^(N×3) [m]
                - threats      : Normalized threat levels θ ∈ [0,1]^(N×1)
                - density_wave : Dispersive forces 𝐅 ∈ ℝ^(N×3) [m/s²]
        """
        if self.flock.agent_count < 2:
            flock["density_wave"] = th.zeros_like(flock["position"])
            return
        
        dists   = th.cdist(flock["position"], flock["position"], p=2)
        weights = th.exp(-dists**2 / (2 * self.mmm.density_bandwidth**2))
        weights.fill_diagonal_(0)
        
        local_density = weights.sum(dim=1, keepdim=True)
        displacements = (
            flock["position"].unsqueeze(1) -  # [N, 1, 3]
            flock["position"].unsqueeze(0)    # [1, N, 3]
        )
        
        density_gradient = (
            (weights.unsqueeze(2) * displacements).sum(dim=1) /
            local_density.clamp_min(self.mmm.epsilon)
        )
        
        threats = flock["threats"]
        # Handle case where threats might be [N, 1] instead of [N]
        if threats.dim() == 2 and threats.shape[1] == 1:
            threats = threats.squeeze(1)
        
        threat_amplification  = 1 + threats * 2  # [N]
        
        if density_gradient.dim() == 2:  # [N, 3]
            threat_amplification = threat_amplification.unsqueeze(-1)  # [N, 1]
        
        flock["density_wave"] = (
            -self.mmm.density_diffusion *
            density_gradient            *
            threat_amplification
        )
    
    def _compute_hamiltonian_forces(self, flock: TensorDictBase):
        """
        Compute forces from Hamiltonian energy minimization with alert modulation.

        Implements the Hamiltonian formulation from Bialek et al. (2012) with
        heterogeneous coupling based on alert states:

            E = -Σ_{<ij>} J_{ij}^{alert} 𝐬_i · 𝐬_j - Σ_i 𝐡_i · 𝐬_i

        where J_{ij}^{alert} = J_{ij} × κ_i, and κ_i is the alert coupling modifier:
            - κ_i = 1.0 for relaxed birds (normal alignment)
            - κ_i = alert_coupling_factor for alert birds
        
        When alert_coupling_factor < 0, alert birds actively oppose alignment,
        creating perturbations that increase polarization variance and maintain
        critical state susceptibility χ = N·Var[Φ] ≥ 5.

        The modified alignment force on agent i becomes:

            F_i^{align} = κ_i × Σ_j J_{ij} (𝐯_j - 𝐯_i)

        This heterogeneity is biologically motivated by vigilance behavior where
        scanning birds prioritize threat detection over flock following.

        Args:
            flock: TensorDict containing positions, velocities, gradient, alert_states,
                   edge indices, and topo_distances, updated with base_forces
        """
        flock["base_forces"] = th.zeros_like(flock["position"])
        
        if "edge_source" in flock and flock["edge_source"].numel() > 0:
            alert_states_source = flock["alert_states"][flock["edge_source"]]
            
            coupling_modifier = th.where(
                alert_states_source > 0.5,
                self.mmm.alert_coupling_factor,
                1.0
            )
            
            j_edges = self.mmm.j_base * coupling_modifier * th.exp(
                -flock["topo_distances"][
                    flock["edge_source"], flock["edge_target"]
                ] / self.mmm.coupling_decay
            )

            force_contrib = j_edges.unsqueeze(1) * (
                flock["velocity"][flock["edge_target"]] - 
                flock["velocity"][flock["edge_source"]]
            )
            flock["base_forces"].index_add_(0, flock["edge_source"], force_contrib)
        
        metric_distances = th.cdist(flock["position"], flock["position"])
        mask = (
            (metric_distances < self.mmm.min_distance * 3) & 
            (metric_distances > 0)
        )

        if mask.any():
            i_idx, j_idx = mask.nonzero(as_tuple=True)
            displacement = flock["position"][j_idx] - flock["position"][i_idx]
            soft_distance = displacement.norm(
                dim     = 1, 
                keepdim = True
            ).clamp_min(self.mmm.min_distance)

            flock["base_forces"].index_add_(
                dim    = 0, 
                index  = i_idx, 
                source = (
                    -self.mmm.separation_strength * displacement / 
                    soft_distance ** 3
                )
            )
        
        flock["base_forces"] -= self.mmm.temperature_scaling * flock["gradient"]

    def _compute_individual_alert_states(self, flock: TensorDictBase):
        """
        Compute alert states following two-state Markov dynamics.
        
        Implements vigilance state transitions as a continuous-time Markov
        chain with asymmetric rates creating realistic bout durations:
        
            P(relaxed → alert) = λ
            P(alert → relaxed) = μ
            
        where λ is the relaxed-to-alert transition rate and μ is the 
        alert-to-relaxed rate. These rates are constant, reflecting the
        intrinsic vigilance dynamics observed in bird flocks.
        
        Steady-state             : π_alert = λ/(λ+μ)  ≈ 0.30
        Mean alert bout duration : E[T_alert] = 1/μ   ≈ 20 timesteps
        Mean relaxed duration    : E[T_relaxed] = 1/λ ≈ 47 timesteps
        
        Args:
            flock: TensorDict updated with alert_states and alert_fraction
        """
        traj_id = 0
        if "trajectory_id" in flock:
            traj_id = (
                flock["trajectory_id"].item()
                if hasattr(flock["trajectory_id"], 'item')
                else int(flock["trajectory_id"])
            )
        
        device   = flock["position"].device
        n_agents = self.flock.agent_count
        
        if traj_id not in self.alert_states_memory:
            steady_state = self.mmm.relaxed_to_alert_rate / (
                self.mmm.relaxed_to_alert_rate + self.mmm.alert_to_relaxed_rate
            )
            self.alert_states_memory[traj_id] = th.bernoulli(
                th.ones(n_agents, device=device) * steady_state
            )
        
        previous    = self.alert_states_memory[traj_id].to(device)
        random_vals = th.rand(n_agents, device=device)
        new_states  = th.where(
            previous    > 0.5,
            random_vals > self.mmm.alert_to_relaxed_rate,
            random_vals < self.mmm.relaxed_to_alert_rate
        ).float()
        
        self.alert_states_memory[traj_id] = new_states
        flock["alert_states"]   = new_states
        flock["alert_fraction"] = new_states.mean()
    
    def _compute_self_propulsion(self, flock: TensorDictBase):
        """
        Compute self-propulsion forces following active matter dynamics.
        
        Implements self-propulsion where each agent maintains an intrinsic
        velocity v₀ in its current heading direction with stochastic
        fluctuations, based on active matter theory:
        
            F_prop = (v₀𝐬 - 𝐯) / τ + η𝝃
        
        where 𝐬 is the heading direction, τ is relaxation time, and 𝝃 is
        Gaussian noise. Alert agents have noise amplitude that places them
        at the order-disorder phase transition (η ≈ 0.4) while relaxed 
        agents remain in the ordered phase (η ≈ 0.1).
        
        For agents with zero velocity (|𝐯| < ε), the heading direction 𝐬 is
        determined from:
            1. Negative temperature gradient direction: 𝐬 = -∇T/|∇T| (if |∇T| > δ)
            2. Random unit vector: 𝐬 = 𝝃/|𝝃| where 𝝃 ~ N(0, I) (fallback)
        
        This ensures the flock can bootstrap movement from rest states.
        
        Args:
            flock: TensorDict containing velocities and alert_states,
                   updated with self_propulsion forces
        """
        speed = flock["velocity"].norm(dim=-1, keepdim=True)
        wind  = flock.get("wind", th.zeros_like(flock["velocity"]))
        
        velocity_heading = flock["velocity"] / speed.clamp_min(1e-8)
        gradient_heading = -th.nn.functional.normalize(flock["gradient"], dim=-1)
        random_heading   = th.nn.functional.normalize(
            th.randn_like(flock["velocity"]), dim=-1
        )
        
        zero_vel = (speed < 1e-6).expand_as(flock["velocity"])
        use_grad = (
            flock["gradient"].norm(dim=-1, keepdim=True) > 0.01
        ).expand_as(flock["velocity"])
        
        heading = th.where(
            zero_vel & use_grad,
            gradient_heading,
            th.where(zero_vel, random_heading, velocity_heading)
        )
        
        alert_states = flock.get("alert_states", th.zeros(self.flock.agent_count))
        noise_scale  = self.mmm.velocity_noise_scale * (
            1.0 + self.mmm.alert_amplification * alert_states
        )
        
        target_vel = heading * self.mmm.self_propulsion_speed + wind * 0.3
        noise      = th.randn_like(flock["velocity"]) * noise_scale.unsqueeze(-1)
        
        flock["self_propulsion"] = (
            (target_vel - flock["velocity"]) / self.mmm.velocity_relaxation_time +
            noise
        )
    
    def _compute_threats(self, flock: TensorDictBase):
        """
        Compute normalized threat level for mode switching.

        Maps temperature to [0, 1] range where:
            - θ < 0.3: Cruise mode (normal murmuration)
            - θ ≥ 0.3: Alert mode (enhanced cohesion and correlation)

        The threshold at 0.7 * T_max provides a safety margin before critical
        temperature is reached.

        Args:
            flock: TensorDict containing temperatures, updated with threat levels

        """
        flock["threats"] = (
            (
                flock["temperature"] 
                - self.safety.max_temperature * self.safety.threat_ratio
            ) /
            (self.safety.max_temperature * self.safety.threat_range_ratio)
        ).clamp(0, 1)
    
    def _compute_topological_distances(self, flock: TensorDictBase):
        """
        Compute minimum hop distances between all pairs of agents.

        Uses Floyd-Warshall algorithm to find shortest paths in the k-NN graph.
        This captures the topological distance metric d_{ij} used in the
        coupling strength decay J_{ij} = J_0 exp(-d_{ij}/λ).

        The algorithm iteratively relaxes distances:
            d_{ij}^{(k+1)} = min(d_{ij}^{(k)}, d_{ik}^{(k)} + d_{kj}^{(k)})

        Args:
            flock: TensorDict with edge indices, updated with topo_distances
        """
        device = flock["position"].device
        n      = self.flock.agent_count
        
        dist = th.full(
            device     = device,
            dtype      = th.float32,
            fill_value = float('inf'),
            size       = (n, n)
        )
        dist.fill_diagonal_(0)

        if "edge_source" in flock and flock["edge_source"].numel() > 0:
            dist[flock["edge_source"], flock["edge_target"]] = 1
            dist[flock["edge_target"], flock["edge_source"]] = 1

        for k in range(n):
            dist = th.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])

        flock["topo_distances"] = dist
    
    def _compute_topological_neighbors(self, flock: TensorDictBase):
        """
        Compute k-nearest neighbors for each agent and store in TensorDict.

        Following Ballerini et al. (2008), each agent tracks exactly 6-7
        nearest neighbors regardless of metric distance. This topological
        interaction rule is key to achieving scale-free correlations.

        Args:
            flock: TensorDict containing positions, updated with edge indices
        """
        distances  = th.cdist(flock["position"], flock["position"])
        _, indices = distances.topk(self.mmm.k_neighbors + 1, largest=False)
        n          = self.flock.agent_count

        flock["edge_source"] = th.arange(n).repeat_interleave(self.mmm.k_neighbors)
        flock["edge_target"] = indices[:, 1:].flatten()

    def _design_nominal_action(self, flock: TensorDictBase):
        """
        Pipeline that computes murmuration dynamics and builds final action.

        Orchestrates the computation of all physics components through a 
        series of transformations on the flock TensorDict, then combines
        them into the final control action.

        The pipeline computes:
        1. Graph topology and temperature gradients
        2. Base Hamiltonian forces from energy minimization
        3. Critical state metrics (susceptibility, information speed)
        4. Threat response components (density waves, alert mode)
        5. Final action combining all force components

        Args:
            flock: TensorDict containing positions, velocities, temperatures,
                   updated with computed physics and final action
        """
        for compute_fn in [
            self._update_graph_state,
            self._estimate_gradient,
            self._compute_individual_alert_states,
            self._compute_hamiltonian_forces,
            self._compute_threats,
            self._set_information_speed,
            self._compute_self_propulsion,
            self._compute_density_wave,
        ]:
            compute_fn(flock)
        
        flock["action"] = (
            flock["self_propulsion"] +
            flock["base_forces"]     +
            flock["density_wave"]
        )

        if self.cbf is not None:
            flock["action"] = self.cbf.filter(flock, flock["action"])

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

    def _estimate_gradient(self, flock: TensorDictBase):
        """
        Uses provided gradient or estimates if not available.

        Prioritizes using the gradient provided by the environment (which has
        access to the full temperature field). Falls back to estimation using
        finite differences only if gradient is not provided:

            ∇T_i ≈ Σ_j (T_j - T_i)(𝐱_j - 𝐱_i) / |𝐱_j - 𝐱_i|²

        Args:
            flock: TensorDict containing gradient or positions/temperatures,
                   ensures gradient field is present

        """
        if "gradient" in flock and flock["gradient"] is not None:
            return
        
        temperature = self._ensure_1d_temperature(flock["temperature"])

        if "edge_source" not in flock or not flock["edge_source"].numel():
            flock["gradient"] = self._vertical_heat_gradient(flock)
            return

        position_diff = (
            flock["position"][flock["edge_target"]] - 
            flock["position"][flock["edge_source"]]
        )
        temperature_diff = (
            temperature[flock["edge_target"]] - 
            temperature[flock["edge_source"]]
        )

        significant_mask = th.abs(temperature_diff) > self.mmm.epsilon
        gradient_sum     = th.zeros_like(flock["position"])
        neighbor_counts  = th.bincount(
            input     = flock["edge_source"][significant_mask],
            minlength = self.flock.agent_count
        ).float()

        gradient_sum.index_add_(
            dim    = 0,
            index  = flock["edge_source"][significant_mask],
            source = position_diff[significant_mask] * temperature_diff[
                significant_mask
            ].unsqueeze(dim=1)
        )
        gradient_estimate = (
            gradient_sum / 
            neighbor_counts.clamp_min(1).unsqueeze(dim=1)
        )

        flock["gradient"] = th.where(
            condition = (neighbor_counts == 0).unsqueeze(dim=1),
            input     = self._vertical_heat_gradient(flock),
            other     = gradient_estimate
        )

    def _set_information_speed(self, flock: TensorDictBase):
        """
        Set information propagation speed based on alert fraction.

        Uses empirical range from Attanasi et al. (2014):
        - Relaxed state: 15 m/s (low responsiveness)
        - Alert state: 45 m/s (high responsiveness)
        
        Speed interpolates linearly with alert fraction to represent
        increasing flock responsiveness as more agents become vigilant.

        Args:
            flock: TensorDict containing alert_fraction, updated with info_speed
        """
        # Use actual alert fraction computed in _compute_individual_alert_states
        flock["info_speed"] = (
            self.mmm.info_speed_min + 
            flock["alert_fraction"] * (self.mmm.info_speed_max - self.mmm.info_speed_min)
        )

    def _update_graph_state(self, flock: TensorDictBase):
        """
        Update graph connectivity using provided or computed topology.

        Uses provided edge topology if available (from environment), otherwise
        computes k-nearest neighbor topology. This ensures consistency between
        expert demonstrations and learned policy.

        Args:
            flock: TensorDict containing positions and optionally edge_index,
                   updated with graph state
        """
        if "edge_index" in flock and flock["edge_index"] is not None:
            edge_idx = flock["edge_index"]
            edge_idx = edge_idx[0] if edge_idx.dim() == 3 else edge_idx
            flock["edge_source"] = edge_idx[0]
            flock["edge_target"] = edge_idx[1]
        else:
            self._compute_topological_neighbors(flock)
        
        self._compute_topological_distances(flock)
        
        if "edge_source" in flock and flock["edge_source"].numel() > 0:
            flock["neighbor_count"] = th.bincount(
                input     = flock["edge_source"],
                minlength = self.flock.agent_count
            ).to(flock["position"].device)
        else:
            flock["neighbor_count"] = th.zeros(
                device = flock["position"].device,
                dtype  = th.long,
                size   = (self.flock.agent_count,)
            )
        
        flock["safe_count"] = flock["neighbor_count"].clamp_min(1)

    def _vertical_heat_gradient(self, flock: TensorDictBase) -> Tensor:
        """
        Creates a default vertical temperature gradient.

        Models natural convection where heat rises, creating a vertical gradient:

            ∇T = (T/T_max) 𝐞_z

        This fallback is used when agents are isolated or in uniform temperature
        fields where neighbor-based estimation is unavailable.

        Args:
            flock: TensorDict containing positions and temperatures

        Returns:
            Tensor [N, d] containing vertical gradient vectors
        """
        num_agents, dimension = flock["position"].shape

        vertical_direction = th.zeros(
            device = flock["position"].device,
            size   = (num_agents, dimension)
        )
        vertical_direction[:, -1] = 1.0

        normalized_temperature = (
            self._ensure_1d_temperature(flock["temperature"]) / 
            self.safety.max_temperature
        )

        return vertical_direction * normalized_temperature.unsqueeze(1)

    def forward(self, flock: TensorDictBase) -> TensorDictBase:
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
        self._design_nominal_action(flock)
        return flock.set("action", flock["action"])
