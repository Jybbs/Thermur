"""
Implements murmuration dynamics with topological interactions.

This module provides a biologically-inspired controller based on starling
murmurations, using topological neighborhoods (k-nearest neighbors) rather
than metric distances. The flock maintains critical state dynamics for
rapid information propagation and exhibits distinct cruise/alert modes.
"""
from __future__           import annotations
from torch_geometric.data import Data
from typing               import TYPE_CHECKING

import torch as th

if TYPE_CHECKING:
    from ..environment                import TrajectoryGenerator
    from .safety                      import CBFSafetyFilter
    from config.imitation.controller  import MurmurationModel, SafetyModel
    from torch                        import Tensor


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
        mmm    : MurmurationModel,
        safety : SafetyModel
    ):
        """
        Initializes the controller with the necessary configuration models.

        Args:
            cbf    : Optional Control Barrier Function filter for safety.
                     If None, no safety filtering is applied
            mmm    : Murmuration model with dynamics and weight parameters
            safety : Safety configuration with thresholds and CBF parameters
        """
        super().__init__()
        self.cbf    = cbf
        self.mmm    = mmm
        self.safety = safety
        self.alert_states_memory = {}

    def _compute_density_wave(self, flock: Data):
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
            flock: Data with position 𝐱 ∈ ℝ^(N×3), threats θ ∈ [0,1]^N,
                   updated with density_wave forces 𝐅 ∈ ℝ^(N×3) [m/s²]
        """
        dists   = th.cdist(flock.position, flock.position, p=2)
        weights = th.exp(-dists**2 / (2 * self.mmm.density_bandwidth**2))
        weights.fill_diagonal_(0)
        
        local_density = weights.sum(dim=1, keepdim=True)
        displacements = (
            flock.position.unsqueeze(1) -  # [N, 1, 3]
            flock.position.unsqueeze(0)    # [1, N, 3]
        )
        
        density_gradient = (
            (weights.unsqueeze(2) * displacements).sum(dim=1) /
            local_density.clamp_min(self.mmm.epsilon)
        )
        
        threat_amplification = (1 + flock.threats * 2)
        
        flock.density_wave = (
            -self.mmm.density_diffusion *
            density_gradient            *
            threat_amplification
        )
    
    def _compute_hamiltonian_forces(self, flock: Data):
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
            flock: Data with positions, velocities, gradient, alert_states,
                   edge indices, and hops matrix, updated with base_forces
        """
        flock.base_forces   = th.zeros_like(flock.position) 
        alert_states_source = flock.alert_states[flock.edge_source]
        
        coupling_modifier = th.where(
            alert_states_source > 0.5,
            self.mmm.alert_coupling_factor,
            1.0
        )
        
        j_edges = self.mmm.j_base * coupling_modifier * th.exp(
            -flock.hops[flock.edge_source, flock.edge_target] / 
            self.mmm.coupling_decay
        )

        force_contrib = j_edges.unsqueeze(1) * (
            flock.velocity[flock.edge_target] - 
            flock.velocity[flock.edge_source]
        )
        flock.base_forces.index_add_(0, flock.edge_source, force_contrib)
        
        metric_distances = th.cdist(flock.position, flock.position)
        mask = (
            (metric_distances < self.mmm.min_distance * 3) & 
            (metric_distances > 0)
        )

        if mask.any():
            i_idx, j_idx = mask.nonzero(as_tuple=True)
            displacement = flock.position[j_idx] - flock.position[i_idx]
            soft_distance = displacement.norm(
                dim     = 1, 
                keepdim = True
            ).clamp_min(self.mmm.min_distance)

            flock.base_forces.index_add_(
                dim    = 0, 
                index  = i_idx, 
                source = (
                    -self.mmm.separation_strength * displacement / 
                    soft_distance ** 3
                )
            )
        
        flock.base_forces -= self.mmm.temperature_scaling * flock.gradient

    def _compute_hops(self, flock: Data):
        """
        Compute minimum hop counts between all agent pairs.

        Uses Floyd-Warshall to find shortest paths through the k-NN graph,
        capturing the topological distance d_{ij} for coupling decay:
        
            J_{ij} = J_0 exp(-d_{ij}/λ)

        The algorithm iteratively relaxes hop counts:
        
            d_{ij}^{(k+1)} = min(d_{ij}^{(k)}, d_{ik}^{(k)} + d_{kj}^{(k)})

        Args:
            flock: Data with edge indices, updated with hops matrix
        """
        hops = th.full(
            device     = flock.position.device,
            dtype      = th.float32,
            fill_value = float('inf'),
            size       = (self.mmm.agent_count, self.mmm.agent_count)
        )
        hops.fill_diagonal_(0)

        hops[flock.edge_source, flock.edge_target] = 1
        hops[flock.edge_target, flock.edge_source] = 1

        for k in range(self.mmm.agent_count):
            hops = th.minimum(hops, hops[:, k:k+1] + hops[k:k+1, :])

        flock.hops = hops

    def _compute_individual_alert_states(self, flock: Data):
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
            flock: Data updated with alert_states and alert_fraction
        """
        traj_id = 0
        device  = flock.position.device
        
        if traj_id not in self.alert_states_memory:
            steady_state = self.mmm.relaxed_to_alert_rate / (
                self.mmm.relaxed_to_alert_rate + self.mmm.alert_to_relaxed_rate
            )
            self.alert_states_memory[traj_id] = th.bernoulli(
                th.ones(self.mmm.agent_count, device=device) * steady_state
            )
        
        previous    = self.alert_states_memory[traj_id].to(device)
        random_vals = th.rand(self.mmm.agent_count, device=device)
        new_states  = th.where(
            previous    > 0.5,
            random_vals > self.mmm.alert_to_relaxed_rate,
            random_vals < self.mmm.relaxed_to_alert_rate
        ).float()
        
        self.alert_states_memory[traj_id] = new_states
        flock.alert_states   = new_states
        flock.alert_fraction = new_states.mean()
    
    def _compute_self_propulsion(self, flock: Data):
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
            flock: Data with velocities and alert_states, updated with
                   self_propulsion forces
        """
        speed = flock.velocity.norm(dim=-1, keepdim=True)
        wind  = getattr(flock, "wind", th.zeros_like(flock.velocity))
        
        velocity_heading = flock.velocity / speed.clamp_min(1e-8)
        gradient_heading = -th.nn.functional.normalize(flock.gradient, dim=-1)
        random_heading   = th.nn.functional.normalize(
            th.randn_like(flock.velocity), dim=-1
        )
        
        zero_vel = (speed < 1e-6).expand_as(flock.velocity)
        use_grad = (
            flock.gradient.norm(dim=-1, keepdim=True) > 0.01
        ).expand_as(flock.velocity)
        
        heading = th.where(
            zero_vel & use_grad,
            gradient_heading,
            th.where(zero_vel, random_heading, velocity_heading)
        )
        
        alert_states = flock.alert_states
        noise_scale  = self.mmm.velocity_noise_scale * (
            1.0 + self.mmm.alert_amplification * alert_states
        )
        
        target_vel = heading * self.mmm.self_propulsion_speed + wind * 0.3
        noise      = th.randn_like(flock.velocity) * noise_scale.unsqueeze(-1)
        
        flock.self_propulsion = (
            (target_vel - flock.velocity) / self.mmm.velocity_relaxation_time +
            noise
        )
    
    def _compute_threats(self, flock: Data):
        """
        Compute normalized threat level for mode switching.

        Maps temperature to [0, 1] range where:
            - θ < 0.3: Cruise mode (normal murmuration)
            - θ ≥ 0.3: Alert mode (enhanced cohesion and correlation)

        The threshold at 0.7 * T_max provides a safety margin before critical
        temperature is reached.

        Args:
            flock: Data with temperatures, updated with threat levels

        """
        flock.threats = (
            (
                flock.temperature 
                - self.safety.max_temperature * self.safety.threat_onset_ratio
            ) /
            (self.safety.max_temperature * self.safety.threat_transition_width)
        ).clamp(0, 1)
    
    def _design_nominal_action(self, flock: Data):
        """
        Pipeline that computes murmuration dynamics and builds final action.

        Orchestrates the computation of all physics components through a
        series of transformations on the flock state, then combines them
        into the final control action.

        The pipeline computes:
        1. Graph topology and temperature gradients
        2. Base Hamiltonian forces from energy minimization
        3. Critical state metrics (susceptibility, information speed)
        4. Threat response components (density waves, alert mode)
        5. Final action combining all force components

        Args:
            flock: Data containing positions, velocities, temperatures,
                   updated with computed physics and final action
        """
        for compute_fn in [
            self._update_graph_state,
            self._compute_individual_alert_states,
            self._compute_hamiltonian_forces,
            self._compute_threats,
            self._compute_self_propulsion,
            self._compute_density_wave,
        ]:
            compute_fn(flock)
        
        flock.action = (
            flock.self_propulsion +
            flock.base_forces     +
            flock.density_wave
        )

        if self.cbf is not None:
            flock.action = self.cbf.filter(flock, flock.action)

    def _update_graph_state(self, flock: Data):
        """
        Update graph connectivity from provided edge topology.

        Extracts edge_source and edge_target from the edge_index tensor
        and computes hop distances for the Hamiltonian forces.

        Args:
            flock: Data containing edge_index from TrajectoryGenerator
        """
        assert flock.edge_index is not None
        flock.edge_source = flock.edge_index[0]
        flock.edge_target = flock.edge_index[1]
        
        self._compute_hops(flock)

    def forward(self, flock: Data) -> Tensor:
        """
        Compute expert control actions from PyG Data flock state.
        
        Orchestrates the full murmuration dynamics pipeline including
        Hamiltonian forces, density waves, threat response, and thermal
        gradient following.
        
        Args:
            flock: PyG Data with position, velocity, temperature, gradient,
                   wind, and edge_index
            
        Returns:
            Control actions (accelerations) [N, 3]
        """
        self._design_nominal_action(flock)
        return flock.action
    
    def generate_trajectories(
        self,
        generator     : TrajectoryGenerator,
        num_timesteps : int
    ) -> list[Data]:
        """
        Generate expert demonstration trajectory as PyG Data objects.
        
        Produces a sequence of graph snapshots representing the flock's evolution
        under expert control. Each timestep captures the full state (positions,
        velocities, environmental data) and the expert's action, creating a dataset
        suitable for behavioral cloning.
        
        The feature vector for each agent concatenates:
        - Position             (3D)
        - Velocity             (3D) 
        - Temperature          (1D)
        - Temperature gradient (3D)
        - Wind field           (3D)
        
        This 13-dimensional representation matches the GNN policy's expected input.
        
        Args:
            generator     : Trajectory generator providing physics simulation
            num_timesteps : Number of simulation steps to generate
        
        Returns:
            List of PyG Data objects, one per timestep, containing:
                - action       : Expert control actions     [N, 3]
                - alert_states : Binary vigilance states    [N]
                - edge_index   : Topological connectivity   [2, E]
                - gradient     : Temperature gradient       [N, 3]
                - position     : Agent positions            [N, 3]
                - temperature  : Temperature values         [N, 1]
                - timestep     : Temporal index
                - velocity     : Agent velocities           [N, 3]
                - wind         : Wind field                 [N, 3]
                - x            : Concatenated node features [N, 13]
        """
        self.alert_states_memory.clear()
        state      = generator.reset()
        trajectory = []
        
        for t in range(num_timesteps):
            action = self.forward(state)
            
            features = th.cat(
                dim     = -1,
                tensors = [
                    state.position,
                    state.velocity,
                    state.temperature,
                    state.gradient,
                    state.wind
                ]
            )
            
            trajectory.append(
                Data(
                    action       = action,
                    alert_states = state.alert_states,
                    edge_index   = state.edge_index,
                    gradient     = state.gradient,
                    position     = state.position,
                    temperature  = state.temperature,
                    timestep     = t,
                    velocity     = state.velocity,
                    wind         = state.wind,
                    x            = features
                )
            )
            
            state = generator.step(action)
        
        return trajectory
