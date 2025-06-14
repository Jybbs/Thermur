"""
Implements the Control Barrier Function (CBF) based safety filter.

This module is responsible for ensuring that the swarm operates within its
defined safety constraints, specifically the maximum thermal limit. It achieves
this by solving a Quadratic Program (QP) at each timestep using the torch-native
`qpth` library.
"""
from __future__           import annotations
from qpth.qp              import QPFunction
from src.configs.pydantic import SafetyConfig
from src.core.structures  import SwarmData
from torch                import Tensor
import torch


class SafetyFilter:
    """
    A safety layer that uses a CBF to filter unsafe control actions.

    This class formulates and solves the real-time Quadratic Program:
        u* = argmin ||u - u_nom||^2
        s.t.  ∇h(x) @ u >= -α * h(x)

    This ensures the control action `u*` does not lead the agent out of the
    pre-defined safe set `C = {x | h(x) >= 0}`.
    """

    def __init__(self, config: SafetyConfig):
        """
        Initializes the safety filter with its required configurations.

        Unlike OSQP, `qpth` is stateless, so no solver setup is needed here.
        We pre-construct the constant identity matrix `Q` for efficiency.

        Args:
            config: A composite config containing all necessary sub-configs
                    for the agent, swarm, CBF, and QP solver.
        """
        self.config = config
        self.Q = torch.eye(
            out    = self.config.swarm.spatial_dims,
            dtype  = torch.float32,
            device = "cpu",
        )

    def _compute_barrier(self, sd: SwarmData) -> tuple[Tensor, Tensor]:
        """
        Computes the barrier function h(x) and its gradient ∇h(x).

        The barrier function is defined as h(x) = T_max - T(x).

        Args:
            sd: The current observation data for the swarm.

        Returns:
            A tuple containing (h_values, h_grads).
        """
        h_values = self.config.agent.max_temperature - sd.temperature
        h_grads  = -sd.temperature_grad
        return h_values, h_grads

    def filter(
        self, 
        sd        : SwarmData, 
        u_nominal : Tensor
    ) -> Tensor:
        """
        Filters a nominal control action to ensure safety using `qpth`.

        This method translates the CBF safety constraint, `∇h(x) @ u >= -αh(x)`,
        into a batch of standard Quadratic Programs (QPs) and solves them to
        find the safe action `u*` that is minimally distant from the desired
        nominal action `u_nom`.

        The objective `min ||u - u_nom||²` and the CBF inequality are encoded
        into the QP matrices `Q`, `p`, `G`, and `h` for the `qpth` solver.

        Args:
            sd        : The current observation data for the swarm.
            u_nominal : The desired control action from the policy network.

        Returns:
            The batch of safe control actions `u*`.
        """
        h_values, h_grads = self._compute_barrier(sd)
        agent_count       = self.config.swarm.agent_count
        device            = u_nominal.device

        # Instantiate the QP solver with its configuration.
        solver = QPFunction(
            eps     = self.config.qp.eps,
            maxIter = self.config.qp.max_iter,
        )

        try:
            u_safe = solver(
                Q = self.Q.to(device).expand(agent_count, -1, -1),
                p = -u_nominal,
                G = -h_grads.unsqueeze(1),
                h = (self.config.cbf.alpha * h_values).unsqueeze(1),
                A = torch.empty(0, device=device),
                b = torch.empty(0, device=device),
            )

        except Exception as e:
            if self.config.qp.on_failure == "use_nominal":
                return u_nominal
            
            else:
                raise ValueError(f"QP safety filter failed to find a solution: {e}")

        return u_safe.view_as(u_nominal)
