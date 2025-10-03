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
        default     = 50,
        ge          = 8,
        description = (
            "Total number of agents N in the multi-agent system, determining "
            "computational complexity and emergent flock dynamics scale. "
            "Minimum of 8 ensures k-nearest neighbor connectivity with k=7."
        )
    )
    coupling_decay: PositiveFloat = Field(
        default     = 0.75,
        description = (
            "Exponential decay rate λ for topological interaction strength "
            "J_ij = J_0 exp(-d_ij/λ). Value of 0.75 extends influence beyond "
            "immediate neighbors while maintaining local interaction structure "
            "that allows heterogeneous patterns to emerge."
        )
    )
    density_bandwidth: PositiveFloat = Field(
        default     = 10.0,
        description = (
            "Spatial scale σ for Gaussian kernel density estimation in meters, "
            "determining the effective radius of local density calculations."
        )
    )
    density_diffusion: PositiveFloat = Field(
        default     = 0.5,
        description = (
            "Diffusion coefficient D in density wave equation ∂ρ/∂t + ∇·(ρv) = D∇²ρ, "
            "controlling how density perturbations spread through the flock."
        )
    )
    heterogeneity_mean: PositiveFloat = Field(
        default     = 0.7,
        description = (
            "Mean μ of heterogeneous noise distribution η_i ~ N(μ, σ). "
            "Value of 0.7 creates stronger heterogeneity that drives critical "
            "dynamics for murmuration patterns and scale-free correlations."
        )
    )
    heterogeneity_std: PositiveFloat = Field(
        default     = 0.20,
        description = (
            "Standard deviation σ of heterogeneous noise distribution η_i ~ N(μ, σ). "
            "From Guisandez et al. (2018), σ = 0.20 creates continuous phase transitions "
            "with critical exponents β = 0.69, γ = 1.7, ν = 1.56. This heterogeneity "
            "naturally generates the behavioral variance necessary for scale-free "
            "correlations and murmuration patterns without requiring explicit anti-alignment."
        )
    )
    initial_spacing: PositiveFloat = Field(
        default     = 0.75,
        ge          = 0.3,
        description = (
            "Spacing between agents in meters when initializing trajectory positions. "
            "Scales the Fibonacci lattice arrangement used for starting configurations."
        )
    )
    j_base: PositiveFloat = Field(
        default     = 2.5,
        description = (
            "Base coupling strength J_0 in maximum entropy formulation controlling "
            "velocity alignment between neighbors. Value of 2.5 provides moderate "
            "coupling strength, balancing cohesion with flexibility to allow "
            "heterogeneous noise to create variance for critical state dynamics."
        )
    )
    k_neighbors: PositiveInt = Field(
        default     = 7,
        description = (
            "Number of topological nearest neighbors each agent tracks, based on "
            "empirical observations of 6-7 neighbors in real starling flocks."
        )
    )
    marginal_amplitude: PositiveFloat = Field(
        default     = 1.95,
        description = (
            "Amplitude λ for marginal speed confinement quartic potential. "
            "From Cavagna et al. (2022), controls individual speed regulation via "
            "V = (λ/v₀⁶)(|v|² - v₀²)⁴ producing F = -4λ/v₀⁶ · (|v|² - v₀²)³ · v. "
            "Independent parameter with same physical dimensions as J_0."
        )
    )
    min_distance: PositiveFloat = Field(
        default     = 0.1,
        description = (
            "Minimum inter-agent distance ε_dist in meters enforced during separation "
            "force computation to prevent singularities in U_sep = Σ 1/||𝐱_i - 𝐱_j||."
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
    separation_strength: PositiveFloat = Field(
        default     = 0.45,
        description = (
            "Weight coefficient for short-range separation forces that prevent "
            "collisions between agents when metric distance < 3·ε_dist, implementing "
            "F_sep = -w_sep · Σ (𝐱_j - 𝐱_i) / ||𝐱_j - 𝐱_i||³."
        )
    )
    temperature_scaling: PositiveFloat = Field(
        default     = 0.15,
        description = (
            "Multiplicative factor λ_thermal adjusting thermal avoidance strength "
            "relative to murmuration forces, balancing safety versus cohesive behavior."
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
