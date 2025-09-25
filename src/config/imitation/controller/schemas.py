"""
Controller domain schemas for Pydantic validation.

This module consolidates all controller configuration models including
flocking parameters, safety settings, and murmuration dynamics.
"""
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt


class MurmurationModel(BaseModel, extra="forbid"):
    """
    Unified configuration for murmuration dynamics and control weights.

    Combines topological interaction parameters and critical state dynamics
    into a single model that fully specifies the murmuration behavior. The
    controller uses these parameters to implement biologically-inspired
    flocking with scale-free correlations
    and rapid information propagation.

    The murmuration exists at a critical state (phase transition) where
    susceptibility diverges, enabling near-instantaneous response to threats
    while maintaining cohesion through topological neighbor tracking.
    """
    agent_count: PositiveInt = Field(
        default     = 30,
        ge          = 8,
        description = (
            "Total number of agents N in the multi-agent system, determining "
            "computational complexity and emergent flock dynamics scale. "
            "Minimum of 8 ensures k-nearest neighbor connectivity with k=7."
        )
    )
    alert_amplification: PositiveFloat = Field(
        default     = 4.5,
        description = (
            "Amplification factor α for alert state noise. Alert birds experience "
            "noise η_alert = η_base × (1 + α), placing them into the disordered "
            "phase where they create perturbations that propagate through the flock. "
            "Value of 4.5 ensures alert birds operate near the disorder transition, "
            "generating variance that maintains elevated susceptibility."
        )
    )
    alert_coupling_factor: float = Field(
        default     = -1.3,
        ge          = -2.0,
        le          = 1.0,
        description = (
            "Coupling strength modifier for alert birds in Hamiltonian alignment. "
            "J_alert = J_base × alert_coupling_factor. Value of -1.3 means alert "
            "birds actively oppose alignment (negative coupling), creating the "
            "oscillations and variance that maintain critical state dynamics. Based on "
            "vigilance behavior where scanning birds prioritize threat detection "
            "over flock following (Beauchamp 2015, Fernández-Juricic 2012)."
        )
    )
    alert_to_relaxed_rate: PositiveFloat = Field(
        default     = 0.05,
        le          = 1.0,
        description = (
            "Transition rate μ from alert to relaxed state (per timestep). "
            "Mean alert duration = 1/μ = 20 timesteps. Based on vigilance "
            "bout durations observed in birds."
        )
    )
    communication_range: PositiveFloat = Field(
        default     = 50.0,
        description = (
            "Metric interaction radius R_comm in meters defining the spatial scale of "
            "the flock. While murmuration control uses topological k-NN, this radius "
            "sets initial agent spacing and safety boundaries where ||𝐱_i - 𝐱_j|| ≤ R_comm."
        )
    )
    coupling_decay: PositiveFloat = Field(
        default     = 0.3,
        description = (
            "Exponential decay rate λ for topological interaction strength "
            "J_ij = J_0 exp(-d_ij/λ). Value of 0.3 ensures influence is primarily "
            "limited to k-nearest topological neighbors, creating local interactions "
            "that allow heterogeneous patterns to emerge rather than global synchronization."
        )
    )
    density_bandwidth: PositiveFloat = Field(
        default     = 5.0,
        description = (
            "Spatial scale σ for Gaussian kernel density estimation in meters, "
            "determining the effective radius of local density calculations."
        )
    )
    density_diffusion: PositiveFloat = Field(
        default     = 0.1,
        description = (
            "Diffusion coefficient D in density wave equation ∂ρ/∂t + ∇·(ρv) = D∇²ρ, "
            "controlling how density perturbations spread through the flock."
        )
    )
    frames_per_episode: PositiveInt = Field(
        default     = 1000,
        description = (
            "Number of timesteps per demonstration episode. Longer episodes "
            "capture extended temporal dependencies in flocking behavior."
        )
    )
    j_base: PositiveFloat = Field(
        default     = 0.5,
        description = (
            "Base coupling strength J_0 in Hamiltonian formulation controlling "
            "velocity alignment between neighbors. Value of 0.5 balances cohesion "
            "with flexibility, allowing perturbations from alert birds to propagate "
            "through the flock while maintaining structural integrity. This enables "
            "the variance generation needed to maintain critical state dynamics."
        )
    )
    k_neighbors: PositiveInt = Field(
        default     = 7,
        description = (
            "Number of topological nearest neighbors each agent tracks, based on "
            "empirical observations of 6-7 neighbors in real starling flocks."
        )
    )
    min_distance: PositiveFloat = Field(
        default     = 0.1,
        description = (
            "Minimum inter-agent distance ε_dist in meters enforced during separation "
            "force computation to prevent singularities in U_sep = Σ 1/||𝐱_i - 𝐱_j||."
        )
    )
    separation_strength: PositiveFloat = Field(
        default     = 1.5,
        description = (
            "Weight coefficient for short-range separation forces that prevent "
            "collisions between agents when metric distance < 3·ε_dist, implementing "
            "F_sep = -w_sep · Σ (𝐱_j - 𝐱_i) / ||𝐱_j - 𝐱_i||³."
        )
    )
    relaxed_to_alert_rate: PositiveFloat = Field(
        default     = 0.021,
        le          = 1.0,
        description = (
            "Transition rate λ from relaxed to alert state (per timestep). "
            "Steady-state alert fraction = λ/(λ+μ) ≈ 0.30 matching the "
            "~30% vigilance observed in bird flocks."
        )
    )
    self_propulsion_speed: PositiveFloat = Field(
        default     = 11.1,
        description = (
            "Self-propulsion speed v₀ in m/s from active matter theory, representing "
            "the intrinsic cruising speed birds maintain. Empirical observations show "
            "starlings fly at 9-12 m/s during murmuration displays (Ballerini 2008). "
            "11.1 m/s represents a weighted average from empirical data."
        )
    )
    temperature_scaling: PositiveFloat = Field(
        default     = 1.0,
        description = (
            "Multiplicative factor λ_thermal adjusting thermal avoidance strength "
            "relative to murmuration forces, balancing safety versus cohesive behavior."
        )
    )
    total_frames: PositiveInt = Field(
        default     = 200_000,
        description = (
            "Total demonstration frames to generate across all scenarios. "
            "Determines the size of the offline dataset for expert trajectory collection."
        )
    )
    velocity_noise_scale: PositiveFloat = Field(
        default     = 0.2,
        description = (
            "Noise amplitude η for velocity fluctuations in active matter models. "
            "Value of 0.2 places the system near the critical point of the Vicsek "
            "model (η_c ≈ 0.15-0.25), where susceptibility is maximized. Implements "
            "stochastic perturbations via 𝐯' = v₀(𝐬 + η𝝃) where 𝝃 ~ N(0,1)."
        )
    )
    velocity_relaxation_time: PositiveFloat = Field(
        default     = 0.6,
        description = (
            "Time constant τ (in seconds) for velocity relaxation in self-propulsion "
            "dynamics: F_prop = (v_target - v)/τ + η𝝃. Based on active matter theory "
            "(Ginelli, 2016), values of 0.5-2.0s provide responsive yet smooth motion. "
            "Value of 0.6s ensures quick response to perturbations while maintaining "
            "realistic bird flight dynamics."
        )
    )


class SafetyModel(BaseModel, extra="forbid"):
    """
    Unified safety system configuration.

    Configures the thermal safety penalty system using Kreisselmeier-Steinhauser
    soft constraints to ensure all control commands respect thermal limits without
    requiring optimization solvers.
    """
    ks_kappa: PositiveFloat = Field(
        default     = 100.0,
        description = (
            "Penalty weight κ in the KS formulation controlling the magnitude of "
            "safety corrections. Higher values enforce stricter constraint satisfaction "
            "via p = (κ/ρ)·ln(1 + exp(-ρ·c)) where c is the constraint function."
        )
    )
    ks_rho: PositiveFloat = Field(
        default     = 30.0,
        le          = 100.0,
        description = (
            "Sharpness parameter ρ in the KS penalty function determining how closely "
            "the smooth penalty approximates max(0, -c). As ρ → ∞, the soft penalty "
            "converges to hard constraint behavior. Default of 30 provides moderate "
            "enforcement: smooth enough for gradient descent yet sharp enough for "
            "effective constraint satisfaction. Values above 100 may cause numerical "
            "instability."
        )
    )
    max_temperature: PositiveFloat = Field(
        default     = 475.0,
        description = (
            "Critical temperature threshold T_max in Kelvin defining the safety set "
            "C = {𝐱 | T(𝐱) ≤ T_max} enforced across all components."
        )
    )
    thermal_alpha: PositiveFloat = Field(
        default     = 2.5,
        description = (
            "Convergence rate α > 0 for thermal constraints, controlling how aggressively "
            "the system maintains temperature safety via the constraint "
            "∇T(𝐱)ᵀ𝐮 + α·(T_max - T(𝐱)) ≥ 0."
        )
    )
    threat_onset_ratio: PositiveFloat = Field(
        default     = 0.7,
        le          = 1.0,
        description = (
            "Fraction of T_max where threat detection begins, defining the temperature "
            "at which the flock starts transitioning toward alert readiness."
        )
    )
    threat_transition_width: PositiveFloat = Field(
        default     = 0.3,
        le          = 1.0,
        description = (
            "Fraction of T_max over which threat level scales from 0 to 1, controlling "
            "the temperature range for gradual alert mode transition."
        )
    )
