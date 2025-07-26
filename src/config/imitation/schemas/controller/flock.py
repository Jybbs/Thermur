"""
Flocking controller and flock configuration models.

This module defines configurations for the flocking controller including
Reynolds rules weights, numerical parameters, and multi-agent flock properties.
"""
from pydantic import BaseModel, Field, NonNegativeFloat, PositiveFloat, PositiveInt
from typing   import Literal


class ControllerModel(BaseModel, extra="forbid"):
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
    gradient_step: PositiveFloat = Field(
        default     = 0.1,
        description = (
            "Finite difference step size δ in meters for computing thermal "
            "gradients via ∇T ≈ [T(𝐱+δê) - T(𝐱-δê)]/2δ approximation."
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
        gt          = 0,
        description = (
            "Maximum distance R_comm in meters for edge formation in dynamic graph "
            "G_t where (i,j) ∈ E_t iff ||𝐱_i - 𝐱_j|| ≤ R_comm."
        )
    )
    formation_scale_factor: PositiveFloat = Field(
        default     = 0.5,
        gt          = 0,
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
    max_temperature: PositiveFloat = Field(
        default     = 475.0,
        description = (
            "Critical temperature threshold T_max in Kelvin defining the safety set "
            "C = {𝐱 | T(𝐱) ≤ T_max} enforced by Control Barrier Functions."
        )
    )
    spatial_dims: Literal[2, 3] = Field(
        default     = 3,
        description = (
            "Dimensionality d ∈ {2,3} of the workspace ℝ^d, affecting state space "
            "size and computational complexity of collision detection algorithms."
        )
    )
    thermal_time_constant: PositiveFloat = Field(
        default     = 5.0,
        description = (
            "Thermal response time τ in seconds for RC circuit heat model, governing "
            "temperature evolution dynamics via T_core ≈ T_skin - τ·dT_skin/dt."
        )
    )
    
    @property
    def shape(self) -> tuple[int, int]:
        """
        Returns (agent_count, spatial_dims) for common tensor shape operations.
        """
        return (self.agent_count, self.spatial_dims)
    
    @property
    def state_size(self) -> int:
        """
        Total number of scalar values needed to represent all agent states.
        
        Used for flattened state arrays in MuJoCo where positions and velocities
        are stored as contiguous 1D arrays of size agent_count * spatial_dims.
        """
        return self.agent_count * self.spatial_dims