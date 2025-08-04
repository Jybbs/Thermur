"""
Controller domain schemas for Pydantic validation.

This module consolidates all controller configuration models including
flocking parameters, safety settings, and Reynolds rule weights.
"""
from pydantic import BaseModel, Field, NonNegativeFloat, PositiveFloat, PositiveInt
from typing   import Literal


class ExpertModel(BaseModel, extra="forbid"):
    """
    Unified control system configuration for expert flocking behavior.

    Combines Reynolds flocking weights and numerical parameters into
    a single configuration for the expert controller that generates
    demonstrations for imitation learning.

    The controller computes nominal actions 𝐮_nom from the negative
    gradient of a potential function U, where:

        𝐮_nom = -∇_x U(S_t)

    The potential U combines classical Reynolds rules with thermal constraints:

        U = ω_coh · U_coh + ω_sep · U_sep + ω_align · U_align + ω_thermal · U_thermal

    where the individual potentials are:
        - Cohesion:   U_coh   = (1/2) · Σ_j∈N(i) ||𝐱_i - 𝐱_j||²
        - Separation: U_sep   = Σ_j∈N(i) 1/||𝐱_i - 𝐱_j||
        - Alignment:  U_align = (1/2) · Σ_j∈N(i) ||𝐯_i - 𝐯_j||²
        - Thermal:    U_thermal = 1/(T_max - T_i)
    """
    epsilon: PositiveFloat = Field(
        default     = 1e-8,
        description = (
            "Numerical stability constant ε preventing division by zero in "
            "potential gradient calculations, particularly for separation forces."
        )
    )
    min_distance: PositiveFloat = Field(
        default     = 0.1,
        description = (
            "Minimum inter-agent distance ε_dist in meters enforced during separation "
            "force computation to prevent singularities in U_sep = Σ 1/||𝐱_i - 𝐱_j||."
        )
    )
    temperature_scaling: PositiveFloat = Field(
        default     = 1.0,
        description = (
            "Multiplicative factor λ_thermal adjusting thermal avoidance strength "
            "relative to Reynolds forces, balancing safety versus cohesive behavior."
        )
    )
    w_alignment: NonNegativeFloat = Field(
        default     = 0.8,
        description = (
            "Weight ω_align for velocity alignment potential U_align = (1/2)Σ||𝐯_i - 𝐯_j||², "
            "promoting coordinated motion and reducing relative velocities within neighborhoods."
        )
    )
    w_cohesion: NonNegativeFloat = Field(
        default     = 1.0,
        description = (
            "Weight ω_coh for cohesion potential U_coh = (1/2)Σ||𝐱_i - 𝐱_j||², creating "
            "attractive forces toward local neighborhood center of mass."
        )
    )
    w_separation: NonNegativeFloat = Field(
        default     = 1.5,
        description = (
            "Weight ω_sep for separation potential U_sep = Σ 1/||𝐱_i - 𝐱_j||, generating "
            "repulsive forces that increase rapidly as agents approach collision."
        )
    )
    w_thermal: NonNegativeFloat = Field(
        default     = 2.0,
        description = (
            "Weight ω_thermal for thermal potential U_thermal = 1/(T_max - T_i), creating "
            "exponentially stronger avoidance forces as temperature approaches T_max."
        )
    )


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

    This metric-based connectivity contrasts with topological neighborhoods used
    in biological flocks (typically 6-7 nearest neighbors regardless of distance).
    """
    agent_count: PositiveInt = Field(
        default     = 30,
        gt          = 1,
        description = (
            "Total number of agents N in the multi-agent system, determining "
            "computational complexity and emergent swarm dynamics scale."
        )
    )
    communication_range: PositiveFloat = Field(
        default     = 50.0,
        description = (
            "Maximum distance R_comm in meters for edge formation in dynamic graph "
            "G_t where (i,j) ∈ E_t iff ||𝐱_i - 𝐱_j|| ≤ R_comm."
        )
    )
    formation_scale_factor: PositiveFloat = Field(
        default     = 0.5,
        le          = 1,
        description = (
            "Formation density factor γ ∈ (0, 1] scaling initial agent spacing as "
            "γ × R_comm, balancing connectivity versus spatial coverage at startup."
        )
    )
    initial_formation: Literal["cube", "sphere", "random"] = Field(
        default     = "sphere",
        description = (
            "Starting geometric pattern for agent positions, affecting initial graph "
            "topology and convergence dynamics of the flocking controller."
        )
    )
    thermal_time_constant: PositiveFloat = Field(
        default     = 5.0,
        description = (
            "Thermal response time τ in seconds for RC circuit heat model, governing "
            "temperature evolution dynamics via T_core ≈ T_skin - τ·dT_skin/dt."
        )
    )


class SafetyModel(BaseModel, extra="forbid"):
    """
    Unified safety system configuration.

    Combines Control Barrier Function parameters and QP solver settings
    into a single configuration for the safety filtering system that
    ensures all control commands respect thermal constraints.
    """
    cbf_alpha: PositiveFloat = Field(
        default     = 2.5,
        description = (
            "Class-K function parameter α > 0 defining exponential safety convergence "
            "rate via CBF constraint ∇h(x)ᵀu ≥ -αh(x) where h(x) = T_max - T(x)."
        )
    )
    log_violations: bool = Field(
        default     = True,
        description = (
            "Enable logging of thermal safety violations and CBF activations for "
            "debugging controller behavior and monitoring safety-critical events during training."
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
            "'nominal' uses unfiltered input, 'raise' propagates exception for debugging."
        )
    )


class ThresholdsModel(BaseModel, extra="forbid"):
    """
    Safety threshold configuration used across multiple domains.

    Defines critical thresholds for thermal safety and control intervention
    detection that must be consistent across controller, monitoring, and
    safety filter components.
    """
    activation_tolerance: PositiveFloat = Field(
        default     = 3.0,
        description = (
            "Control deviation threshold in m/s for detecting CBF interventions, "
            "used by both safety filter and event logging systems."
        )
    )
    max_temperature: PositiveFloat = Field(
        default     = 475.0,
        description = (
            "Critical temperature threshold T_max in Kelvin defining the safety set "
            "C = {𝐱 | T(𝐱) ≤ T_max} enforced across all components."
        )
    )
