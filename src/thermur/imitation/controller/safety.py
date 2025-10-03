"""
Implements thermal safety constraints using Kreisselmeier-Steinhauser penalties.

This module provides gradient-based thermal safety enforcement without requiring
optimization solvers. Using smooth penalty functions, it guides agents away from
dangerous thermal regions while maintaining differentiability for neural network
training.
"""
from __future__ import annotations
from typing     import TYPE_CHECKING
from torch      import sigmoid

if TYPE_CHECKING:
    from config.imitation.controller import SafetyModel
    from torch                       import Tensor
    from torch_geometric.data        import Data


class ThermalPenalty:
    """
    A thermal safety layer using Kreisselmeier-Steinhauser soft penalties.

    This class implements smooth thermal constraints that prevent agents from
    exceeding temperature limits. Unlike hard constraint methods, this approach
    uses differentiable penalties that are compatible with gradient-based learning.

    The Kreisselmeier-Steinhauser (KS) function provides a smooth approximation
    to max(0, violation) that maintains differentiability everywhere:

        p(c) = (κ/ρ)·ln(1 + exp(-ρ·c))

    where c(𝐱,𝐮) = ∇T(𝐱)ᵀ𝐮 + α·(T_max - T(𝐱)) is the thermal constraint.

    As ρ → ∞, the penalty approaches hard constraint behavior while maintaining
    numerical stability. The gradient ∇p provides smooth correction directions
    that guide actions toward thermal safety.
    """

    def __init__(self, safety: SafetyModel):
        """
        Initializes the thermal penalty layer with configuration parameters.

        Args:
            safety : Safety configuration with KS parameters and temperature limits
        """
        self.safety          = safety
        self.total_queries   = 0
        self.violation_count = 0

    def filter(
        self,
        flock     : Data,
        u_nominal : Tensor
    ) -> Tensor:
        """
        Applies thermal safety penalties to nominal control actions.

        Uses the Kreisselmeier-Steinhauser function to create smooth penalties
        for constraint violations. The gradient of this penalty provides a
        correction direction that guides actions toward safety.

        The thermal constraint is:

            c(𝐱,𝐮) = ∇T(𝐱)ᵀ𝐮 + α·(T_max - T(𝐱))

        where c ≥ 0 represents thermally safe actions. When c < 0 (violation),
        the KS penalty activates smoothly via:

            ∇p/∇𝐮 = κ·σ(-ρ·c)·∇T

        where σ is the sigmoid function. The correction is then:

            𝐮_safe = 𝐮_nom - ∇p/∇𝐮

        The sigmoid ensures smooth activation:
            - As c → -∞ (larger violations): σ(-ρ·c) → 1, applying full correction
            - As c → 0⁺ (safe region): σ(-ρ·c) → 0, applying no correction

        Args:
            flock     : Current observation data containing gradient ∇T and
                        temperature T tensors
            u_nominal : Desired control actions from the policy network

        Returns:
            Batch of safety-adjusted control actions
        """
        barrier_term = (
            self.safety.thermal_alpha
            * (self.safety.max_temperature - flock.temperature.squeeze())
        )

        constraint = (flock.gradient * u_nominal).sum(dim=1) + barrier_term
        if constraint.any():
            self.violation_count += (constraint < 0).sum().item()
            self.total_queries   += flock.position.shape[0]

            return u_nominal - (
                self.safety.ks_kappa * flock.gradient
                * sigmoid(-self.safety.ks_rho * constraint).unsqueeze(1)
            )

        return u_nominal

    def get_violation_rate(self) -> float:
        """
        Returns the percentage of queries where thermal constraints were violated.

        This metric helps monitor safety performance during training. Lower rates
        indicate better constraint satisfaction.

        Returns:
            Violation rate as a percentage [0, 100]
        """
        return (
            0.0 if self.total_queries == 0
            else (self.violation_count / self.total_queries) * 100.0
        )