"""
Implements the Control Barrier Function (CBF) based safety filter.

This module is responsible for ensuring that the flock operates within its
defined safety constraints, specifically the maximum thermal limit. It achieves
this by solving a Quadratic Program (QP) at each timestep using the torch-native
`qpth` library.
"""
from __future__ import annotations
from qpth.qp import QPFunction
from typing  import TYPE_CHECKING

import torch as th

if TYPE_CHECKING:
    from config.imitation.controller import FlockModel, SafetyModel
    from torch                       import Tensor
    from torch_geometric.data        import Data


class CBFSafetyFilter:
    """
    A safety layer that uses a thermal CBF to filter unsafe control actions.

    This class formulates and solves the real-time Quadratic Program:
        u* = argmin ||u - u_nom||²
        s.t.  ∇h(x) · u ≥ -α · h(x)

    This ensures the control action u* does not lead the agent out of the
    pre-defined safe set C = {x | h(x) ≥ 0}, where h(s) = T_max - T(s)
    creates a safety boundary at the maximum survivable temperature.
    """

    def __init__(
        self,
        flock  : FlockModel,
        safety : SafetyModel
    ):
        """
        Initializes the safety filter with thermal barrier configuration.

        Unlike OSQP, `qpth` is stateless, so no solver setup is needed here.
        We pre-construct the constant identity matrix `Q` for efficiency.

        Args:
            flock  : Flock configuration model containing agent properties
            safety : Safety configuration with CBF parameters and thresholds
        """
        self.activation_count     = 0
        self.activation_tolerance = safety.cbf_tolerance
        self.agent_count          = flock.agent_count
        self.max_temperature      = safety.max_temperature
        self.safety               = safety
        self.total_queries        = 0
        self.Q                    = th.eye(3, dtype=th.float32)

    def _log_activation(self, is_active: Tensor):
        """
        Records barrier function activations for debugging and monitoring.

        Args:
            is_active: Boolean tensor indicating which agents had the CBF
                       actively modify their control input.
        """
        self.activation_count += is_active.sum().item()
        self.total_queries    += is_active.shape[0]

    def filter(
        self,
        flock     : Data,
        u_nominal : Tensor
    ) -> Tensor:
        """
        Filters a nominal control action to ensure safety using `qpth`.

        This method translates the CBF safety constraint, ∇h(x) · u ≥ -αh(x),
        into a batch of standard Quadratic Programs (QPs) and solves them to
        find the safe action `u*` that is minimally distant from the desired
        nominal action `u_nom`.

        The objective `min ||u - u_nom||²` and the CBF inequality are encoded
        into the QP matrices `Q`, `p`, `G`, and `h` for the `qpth` solver.

        Args:
            flock     : The current observation data for the flock containing
                        gradient and temperature tensors.
            u_nominal : The desired control action from the policy network.

        Returns:
            The batch of safe control actions `u*`.
        """
        agent_count = self.agent_count
        device      = u_nominal.device
        h_grads     = -flock.gradient
        h_values    = self.max_temperature - flock.temperature
        solver      = QPFunction(
            eps     = self.safety.qp_eps,
            maxIter = self.safety.qp_max_iter,
            verbose = -1  # Suppress warnings
        )

        try:
            
            Q = self.Q.to(device).expand(agent_count, -1, -1)     # [N, 3, 3]
            p = -u_nominal                                        # [N, 3]
            G = -h_grads.unsqueeze(1)                             # [N, 1, 3]
            h = (self.safety.cbf_alpha * h_values).unsqueeze(-1)  # [N, 1]
            A = th.empty(agent_count, 0, 3, device=device)
            b = th.empty(agent_count, 0, device=device)
            
            u_safe = solver(Q, p, G, h, A, b)

            assert u_safe is not None
            delta     = u_safe - u_nominal
            is_active = delta.norm(dim=1) > self.activation_tolerance
            self._log_activation(is_active)
            return u_safe.view_as(u_nominal)

        except Exception:
            return (
                u_nominal if self.safety.qp_on_failure == "nominal"
                else th.zeros_like(u_nominal)
            )

    def get_activation_rate(self) -> float:
        """
        Returns the percentage of queries where the CBF was active.

        Returns:
            The activation rate as a percentage.
        """
        return (
            0.0 if self.total_queries == 0
            else (self.activation_count / self.total_queries) * 100.0
        )
