"""
Controller domain schemas for Pydantic validation.

This module consolidates all controller configuration models including
flocking parameters, safety settings, and Reynolds rule weights.
"""
from pydantic import BaseModel, Field, NonNegativeFloat, PositiveFloat, PositiveInt
from typing   import Literal


class FlockModel(BaseModel, extra="forbid"):
    """
    Unified configuration for the thermal drone flock.

    Combines agent physical properties, collective behavior parameters,
    and spatial settings into a single coherent configuration used across
    simulation, control, and safety components.

    The thermal properties govern agent survivability and safety constraints.
    The maximum temperature T_max defines the Control Barrier Function's safety
    boundary:

        h(𝐱) = T_max - T(𝐱)

    The thermal time constant τ models heat dissipation dynamics using an RC
    thermal circuit analogy, allowing estimation of core temperature from surface
    measurements:

        T_core ≈ T_skin - τ · dT_skin/dt

    The communication range R_comm determines the dynamic neighborhood graph
    G_t = (V, E_t) at each timestep t, where edges exist between agents i and j
    when:

        ||𝐱_i - 𝐱_j|| ≤ R_comm

    Note: With murmuration dynamics, this metric-based connectivity is overridden
    by topological neighborhoods (k-nearest neighbors) for control calculations,
    though R_comm still affects simulation and safety constraints.
    """
    agent_count: PositiveInt = Field(
        default     = 30,
        gt          = 1,
        description = (
            "Total number of agents N in the multi-agent system, determining "
            "computational complexity and emergent flock dynamics scale."
        )
    )
    communication_range: PositiveFloat = Field(
        default     = 50.0,
        description = (
            "Maximum distance R_comm in meters for edge formation in dynamic graph "
            "G_t where (i,j) ∈ E_t iff ||𝐱_i - 𝐱_j|| ≤ R_comm."
        )
    )
    thermal_time_constant: PositiveFloat = Field(
        default     = 5.0,
        description = (
            "Thermal response time τ in seconds for RC circuit heat model, governing "
            "temperature evolution dynamics via T_core ≈ T_skin - τ·dT_skin/dt."
        )
    )


class MurmurationModel(BaseModel, extra="forbid"):
    """
    Unified configuration for murmuration dynamics and control weights.
    
    Combines topological interaction parameters, critical state dynamics,
    and Reynolds rule weights into a single model that fully specifies
    the murmuration behavior. The controller uses these parameters to
    implement biologically-inspired flocking with scale-free correlations
    and rapid information propagation.
    
    The murmuration exists at a critical state (phase transition) where
    susceptibility diverges, enabling near-instantaneous response to threats
    while maintaining cohesion through topological neighbor tracking.
    """
    alert_threshold: PositiveFloat = Field(
        default     = 0.3,
        description = (
            "Normalized threat level θ_alert ∈ [0,1] triggering transition from cruise "
            "to alert mode, where 0 represents ambient temperature and 1 represents "
            "T_max."
        )
    )
    correlation_exponent: PositiveFloat = Field(
        default     = 0.333,
        description = (
            "Target power-law exponent γ ≈ 1/3 for velocity correlation decay "
            "C(r) ∼ r^(-γ), matching empirical observations of scale-free "
            "correlations in starling flocks."
        )
    )
    correlation_strength: PositiveFloat = Field(
        default     = 1.5,
        description = (
            "Additional alignment weight α_corr applied in alert mode to enhance "
            "velocity "
            "correlation and create tighter, more responsive collective motion."
        )
    )
    coupling_decay: PositiveFloat = Field(
        default     = 0.5,
        description = (
            "Exponential decay rate λ for topological interaction strength "
            "J_ij = J_0 exp(-d_ij/λ), "
            "controlling how influence diminishes with topological distance."
        )
    )
    density_diffusion: PositiveFloat = Field(
        default     = 0.1,
        description = (
            "Diffusion coefficient D in density wave equation ∂ρ/∂t + ∇·(ρv) = D∇²ρ, "
            "controlling how density perturbations spread through the flock."
        )
    )
    density_strength: PositiveFloat = Field(
        default     = 0.8,
        description = (
            "Additional cohesion weight β_dense applied in alert mode to increase "
            "flock "
            "density, creating the characteristic 'ink-like' appearance during evasion."
        )
    )
    epsilon: PositiveFloat = Field(
        default     = 1e-8,
        description = (
            "Numerical stability constant ε preventing division by zero in "
            "potential gradient calculations, particularly for separation forces."
        )
    )
    j_base: PositiveFloat = Field(
        default     = 1.0,
        description = (
            "Base coupling strength J_0 in Hamiltonian formulation, controlling the "
            "overall strength of velocity alignment interactions between neighbors."
        )
    )
    k_neighbors: PositiveInt = Field(
        default     = 7,
        description = (
            "Number of topological nearest neighbors each agent tracks, based on "
            "empirical observations of 6-7 neighbors in real starling flocks."
        )
    )
    info_speed_coefficient: PositiveFloat = Field(
        default     = 30.0,
        description = (
            "Coefficient c_0 for information propagation speed "
            "v_info = c_0 * sqrt(χ/m_eff), "
            "calibrated to achieve empirical range of 15-45 m/s in starling flocks."
        )
    )
    min_distance: PositiveFloat = Field(
        default     = 0.1,
        description = (
            "Minimum inter-agent distance ε_dist in meters enforced during separation "
            "force computation to prevent singularities in U_sep = Σ 1/||𝐱_i - 𝐱_j||."
        )
    )
    susceptibility_amplification: PositiveFloat = Field(
        default     = 2.0,
        description = (
            "Amplification factor α_χ for alignment weight modulation based on "
            "susceptibility, creating stronger velocity correlation as the flock "
            "approaches critical state."
        )
    )
    susceptibility_target: PositiveFloat = Field(
        default     = 10.0,
        description = (
            "Target susceptibility χ_target for maintaining critical state "
            "dynamics, where "
            "χ = N·Var[Φ] measures the flock's responsiveness to perturbations."
        )
    )
    temperature_scaling: PositiveFloat = Field(
        default     = 1.0,
        description = (
            "Multiplicative factor λ_thermal adjusting thermal avoidance strength "
            "relative to Reynolds forces, balancing safety versus cohesive behavior."
        )
    )
    threat_range_ratio: PositiveFloat = Field(
        default     = 0.3,
        le          = 1.0,
        description = (
            "Fraction of T_max over which threat level scales from 0 to 1, controlling "
            "the temperature range for gradual alert mode transition."
        )
    )
    threat_threshold_ratio: PositiveFloat = Field(
        default     = 0.7,
        le          = 1.0,
        description = (
            "Fraction of T_max where threat detection begins, defining the temperature "
            "at which the flock starts transitioning toward alert readiness."
        )
    )
    w_alignment: NonNegativeFloat = Field(
        default     = 1.0,
        description = (
            "Base weight ω_align for velocity alignment potential "
            "U_align = (1/2)Σ||𝐯_i - 𝐯_j||², "
            "modulated by susceptibility to achieve critical state dynamics."
        )
    )
    w_cohesion: NonNegativeFloat = Field(
        default     = 0.8,
        description = (
            "Weight ω_coh for cohesion potential U_coh = (1/2)Σ||𝐱_i - 𝐱_j||², "
            "slightly "
            "reduced from baseline to account for topological neighborhood effects."
        )
    )
    w_separation: NonNegativeFloat = Field(
        default     = 1.2,
        description = (
            "Weight ω_sep for separation potential U_sep = Σ 1/||𝐱_i - 𝐱_j||, "
            "maintained "
            "at standard level to ensure collision avoidance regardless of formation."
        )
    )
    w_thermal: NonNegativeFloat = Field(
        default     = 2.0,
        description = (
            "Weight ω_thermal for thermal potential U_thermal = 1/(T_max - T_i), "
            "creating "
            "exponentially stronger avoidance forces as temperature approaches T_max."
        )
    )


class SafetyModel(BaseModel, extra="forbid"):
    """
    Unified safety system configuration.

    Combines Control Barrier Function parameters, QP solver settings,
    and critical safety thresholds into a single configuration for the
    safety filtering system that ensures all control commands respect
    thermal constraints.
    """
    cbf_alpha: PositiveFloat = Field(
        default     = 2.5,
        description = (
            "Class-K function parameter α > 0 defining exponential safety convergence "
            "rate via CBF constraint ∇h(x)ᵀu ≥ -αh(x) where h(x) = T_max - T(x)."
        )
    )
    cbf_tolerance: PositiveFloat = Field(
        default     = 3.0,
        description = (
            "Control deviation threshold in m/s for detecting CBF interventions, "
            "used by both safety filter and event logging systems."
        )
    )
    log_violations: bool = Field(
        default     = True,
        description = (
            "Enable logging of thermal safety violations and CBF activations for "
            "debugging controller behavior and monitoring safety-critical events "
            "during training."
        )
    )
    max_temperature: PositiveFloat = Field(
        default     = 475.0,
        description = (
            "Critical temperature threshold T_max in Kelvin defining the safety set "
            "C = {𝐱 | T(𝐱) ≤ T_max} enforced across all components."
        )
    )
    qp_eps: PositiveFloat = Field(
        default     = 1e-6,
        description = (
            "Convergence tolerance ε for the quadratic program solver determining when "
            "||u^(k+1) - u^(k)|| < ε indicates optimal solution found."
        )
    )
    qp_max_iter: PositiveInt = Field(
        default     = 100,
        description = (
            "Maximum solver iterations before termination, balancing solution quality "
            "against real-time computational constraints in the control loop."
        )
    )
    qp_on_failure: Literal["zero", "nominal", "raise"] = Field(
        default     = "zero",
        description = (
            "Fallback strategy when QP fails: 'zero' applies zero control for safety, "
            "'nominal' uses unfiltered input, 'raise' propagates exception for "
            "debugging."
        )
    )


