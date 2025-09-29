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
    from ..environment               import TrajectoryGenerator
    from .safety                     import ThermalPenalty
    from config.imitation.controller import MurmurationModel, SafetyModel
    from torch                       import Tensor


class MurmurationController(th.nn.Module):
    """
    Implements murmuration dynamics with topological interactions.

    This controller generates biologically-inspired flocking behavior based on
    starling murmurations, using topological neighborhoods (k-nearest neighbors)
    rather than metric distances. The flock maintains critical state dynamics
    for rapid information propagation through heterogeneous behavioral variance.

    The controller implements the maximum entropy formulation from Bialek et
    al. (2012), who derive an effective energy function for bird flocks using
    statistical inference. This approach infers interaction parameters from
    observed correlations, yielding an energy landscape that captures
    collective behavior without assuming underlying mechanics:

        E = -Σ_{<ij>} J_{ij} 𝐬_i · 𝐬_j - Σ_i 𝐡_i · 𝐬_i

    where:
        - 𝐬_i = 𝐯_i / |𝐯_i| are normalized velocity vectors (spin variables)
        - J_{ij} = J_0 exp(-d_{ij}/λ) with uniform coupling strength
        - d_{ij} is the topological distance (minimum hop count)
        - 𝐡_i represents external fields (thermal gradients)

    Heterogeneous noise η_i ~ N(μ, σ=0.20) following Guisandez et al. (2018)
    creates a spectrum of behavioral responses, wherein agents with low η_i strongly
    align while those with high η_i move independently. 
    
    This variance maintains elevated susceptibility χ ~ N and enables scale-free 
    correlations C(r) ~ r^{-1/3} with information speeds of 15-45 m/s 
    (Attanasi et al. 2014, Cavagna et al. 2010).
    """

    def __init__(
        self,
        mmm     : MurmurationModel,
        penalty : ThermalPenalty,
        safety  : SafetyModel
    ):
        """
        Initializes the controller with the necessary configuration models.

        Args:
            mmm     : Murmuration model with dynamics and weight parameters
            penalty : Thermal safety penalty layer for gradient-based constraints
            safety  : Safety configuration with thresholds and temperature limits
        """
        super().__init__()
        self.cached_edges      = th.empty(0)
        self.cached_hops       = th.empty(0)
        self.heading_rng       = th.Generator()
        self.heterogeneity_rng = th.Generator()
        self.mmm               = mmm
        self.noise_rng         = th.Generator()
        self.penalty           = penalty
        self.safety            = safety

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
        with the effect amplified under threat conditions (high θ). The local
        density is clamped to a minimum of 1e-8 when computing gradients to
        prevent division by zero.

        Args:
            flock: Data with position 𝐱 ∈ ℝ^(N×3), threats θ ∈ [0,1]^N,
                   updated with density_wave forces 𝐅 ∈ ℝ^(N×3) [m/s²]
        """
        weights = th.exp(-flock.distances**2 / (2 * self.mmm.density_bandwidth**2))
        weights.fill_diagonal_(0)

        local_density = weights.sum(dim=1, keepdim=True)
        displacements = (
            flock.position.unsqueeze(1) -  # [N, 1, 3]
            flock.position.unsqueeze(0)    # [1, N, 3]
        )

        density_gradient = (
            (weights.unsqueeze(2) * displacements).sum(dim=1) /
            local_density.clamp_min(1e-8)
        )

        threat_amplification = (1 + flock.threats * 2)

        flock.density_wave = (
            -self.mmm.density_diffusion *
            density_gradient            *
            threat_amplification
        )

    def _compute_energy_forces(self, flock: Data):
        """
        Compute interaction forces from maximum entropy energy function.

        Following Bialek et al. (2012), we use a spin-glass energy formulation
        where normalized velocities act as spins. While not derived from
        canonical mechanics, this generates biologically realistic alignment:

            F_i^{align} = Σ_j J_{ij} (𝐯_j - 𝐯_i)

        with J_{ij} = J_0 exp(-d_{ij}/λ) for topological coupling

        Args:
            flock: Data with edge indices, gradient, hops matrix, positions,
                   and velocities, updated with base_forces
        """
        flock.base_forces = th.zeros_like(flock.position)

        j_edges = self.mmm.j_base * th.exp(
            -flock.hops[flock.edge_source, flock.edge_target] /
            self.mmm.coupling_decay
        )

        force_contrib = j_edges.unsqueeze(1) * (
            flock.velocity[flock.edge_target] -
            flock.velocity[flock.edge_source]
        )
        flock.base_forces.index_add_(0, flock.edge_source, force_contrib)

        mask = (
            (flock.distances < self.mmm.min_distance * 3) &
            (flock.distances > 0)
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
        Compute minimum hop counts between all agent pairs with caching.

        Uses Floyd-Warshall to find shortest paths through the k-NN graph,
        capturing the topological distance d_{ij} for coupling decay:

            J_{ij} = J_0 exp(-d_{ij}/λ)

        The algorithm iteratively relaxes hop counts:

            d_{ij}^{(k+1)} = min(d_{ij}^{(k)}, d_{ik}^{(k)} + d_{kj}^{(k)})

        This implementation caches the result and only recomputes when the
        edge structure changes, avoiding redundant O(N³) computations.

        Args:
            flock: Data with edge indices, updated with hops matrix
        """
        edges = th.stack([flock.edge_source, flock.edge_target])

        if th.equal(self.cached_edges, edges):
            flock.hops = self.cached_hops
            return

        hops = th.full(
            device     = flock.position.device,
            fill_value = float('inf'),
            size       = (self.mmm.agent_count, self.mmm.agent_count)
        )
        hops.fill_diagonal_(0)

        hops[flock.edge_source, flock.edge_target] = 1
        hops[flock.edge_target, flock.edge_source] = 1

        for k in range(self.mmm.agent_count):
            hops = th.minimum(hops, hops[:, k:k+1] + hops[k:k+1, :])

        self.cached_edges = edges
        self.cached_hops  = hops
        flock.hops        = hops

    def _compute_heterogeneity(self, frame: int) -> Tensor:
        """
        Generate heterogeneous behavioral variance deterministically.

        Following Guisandez et al. (2018), individual heterogeneity values are
        drawn from η_i ~ N(μ, σ=0.20). This heterogeneity creates the
        behavioral variance necessary for murmuration patterns:

        - Low η_i  : Strong alignment, follows neighbors closely
        - High η_i : Weak alignment, moves independently

        The variance in responses prevents homogeneous motion and enables
        critical dynamics with continuous phase transitions.

        Args:
            frame: Current simulation frame for deterministic seeding

        Returns:
            Heterogeneity values for each agent [N]
        """
        self.heterogeneity_rng.manual_seed(frame)
        return th.normal(
            generator = self.heterogeneity_rng,
            mean      = self.mmm.heterogeneity_mean,
            size      = (self.mmm.agent_count,),
            std       = self.mmm.heterogeneity_std
        )

    def _compute_self_propulsion(self, flock: Data):
        """
        Compute self-propulsion with marginal speed confinement.

        Implements quartic speed confinement from Cavagna et al. (2022) where
        individual speeds are regulated through potential V = (λ/v₀⁶)(|vᵢ|² - v₀²)⁴
        producing force:

            Fᵢ = -4λ/v₀⁶ · (|vᵢ|² - v₀²)³ · vᵢ

        This marginal confinement:
        - Weakly constrains small deviations (enables natural fluctuations)
        - Strongly suppresses large deviations (enforces biomechanical limits)
        - Preserves scale-free correlations across the flock

        For agents with zero velocity (|v| < ε), the heading direction is
        determined from:
            1. Negative temperature gradient direction: -∇T/|∇T| (if |∇T| > δ)
            2. Random unit vector: ξ/|ξ| where ξ ~ N(0, I) (fallback)

        Args:
            flock: Data with velocities, updated with heterogeneity
                   and self_propulsion forces
        """
        speed   = flock.velocity.norm(dim=-1, keepdim=True)
        heading = flock.velocity / speed.clamp_min(1e-8)

        self.heading_rng.manual_seed(flock.frame)
        random_heading = th.nn.functional.normalize(
            dim   = -1,
            input = th.randn(
                device    = flock.velocity.device,
                generator = self.heading_rng,
                size      = flock.velocity.shape
            )
        )

        is_stationary = (speed < 1e-6).expand_as(flock.velocity)
        has_gradient  = (
            flock.gradient.norm(dim=-1, keepdim=True) > 0.01
        ).expand_as(flock.velocity)

        heading = th.where(
            is_stationary & has_gradient,
            -th.nn.functional.normalize(flock.gradient, dim=-1),
            th.where(is_stationary, random_heading, heading)
        )

        speed_normalized = speed / self.mmm.self_propulsion_speed
        deviation_cubed  = (speed_normalized ** 2 - 1) ** 3
        lambda_effective = self.mmm.j_base * self.mmm.speed_regulation_ratio
        speed_force      = (
            -4 * lambda_effective * deviation_cubed *
            speed_normalized * self.mmm.self_propulsion_speed
        )

        self.noise_rng.manual_seed(flock.frame)
        noise_direction = th.nn.functional.normalize(
            dim   = -1,
            input = th.randn(
                device    = flock.velocity.device,
                generator = self.noise_rng,
                size      = flock.velocity.shape
            )
        )
        flock.heterogeneity   = self._compute_heterogeneity(flock.frame)
        flock.self_propulsion = (
            heading * speed_force
            + noise_direction * flock.heterogeneity.unsqueeze(-1)
            + flock.wind      * self.mmm.wind_coupling
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
        2. Alignment forces from maximum entropy energy minimization
        3. Critical state metrics (susceptibility, information speed)
        4. Threat response components (density waves, alert mode)
        5. Final action combining all force components

        Args:
            flock: Data containing positions, velocities, temperatures,
                   updated with computed physics and final action
        """
        for compute_fn in [
            self._update_graph_state,
            self._compute_energy_forces,
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

        flock.action = self.penalty.filter(flock, flock.action)

    def _update_graph_state(self, flock: Data):
        """
        Update graph connectivity from provided edge topology.

        Extracts edge_source and edge_target from the edge_index tensor
        and computes hop distances for the alignment forces.

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
        maximum entropy alignment forces, density waves, threat response,
        and thermal gradient following.

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
        generator  : TrajectoryGenerator,
        num_frames : int
    ) -> list[Data]:
        """
        Generate expert demonstration trajectory as PyG Data objects.

        Produces a sequence of graph states representing the flock's evolution
        under expert control. Each frame captures the full state (positions,
        velocities, environmental data) and the expert's action, creating a dataset
        suitable for behavioral cloning.

        The feature vector for each agent concatenates:
        - Position             (3D)
        - Velocity             (3D)
        - Temperature          (1D)
        - Temperature gradient (3D)
        - Wind field           (3D)

        This 13-dimensional representation matches the GNN policy's expected input.

        Action tensors are cloned when constructing trajectory states to preserve
        the computed values at each frame. Since the controller modifies the state
        object in-place by adding an action attribute, cloning ensures each saved
        state maintains an independent copy of its action values.

        Args:
            generator  : Trajectory generator providing physics simulation
            num_frames : Number of simulation frames to generate

        Returns:
            List of PyG Data objects, one per frame, containing:
                - action        : Expert control actions      [N, 3]
                - edge_index    : Topological connectivity    [2, E]
                - frame         : Temporal index
                - gradient      : Temperature gradient        [N, 3]
                - heterogeneity : Individual noise amplitudes [N]
                - position      : Agent positions             [N, 3]
                - temperature   : Temperature values          [N, 1]
                - velocity      : Agent velocities            [N, 3]
                - wind          : Wind field                  [N, 3]
                - x             : Concatenated node features  [N, 13]
        """
        state      = generator.reset()
        trajectory = []

        for frame in range(num_frames):
            state.frame = frame
            action      = self.forward(state)
            features    = th.cat(
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
                    action        = action,
                    edge_index    = state.edge_index,
                    frame         = frame,
                    gradient      = state.gradient,
                    heterogeneity = state.heterogeneity,
                    position      = state.position,
                    temperature   = state.temperature,
                    velocity      = state.velocity,
                    wind          = state.wind,
                    x             = features
                )
            )

            state = generator.step(action)

        return trajectory
