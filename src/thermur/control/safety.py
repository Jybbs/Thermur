"""
Implements the Control Barrier Function (CBF) based safety filter.

This module is responsible for ensuring that the flock operates within its
defined safety constraints, specifically the maximum thermal limit. It achieves
this by solving a Quadratic Program (QP) at each timestep using the torch-native
`qpth` library.
"""
from __future__        import annotations
from configs.imitation import SafetyModel
from qpth.qp           import QPFunction
from tensordict        import TensorDict
from torch             import Tensor

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

    def __init__(
        self, 
        agent_count  : int,
        barrier      : ThermalBarrierFunction,
        cbf_alpha    : float,
        safety       : SafetyModel,
        spatial_dims : int,
    ):
        """
        Initializes the safety filter with its required configurations.

        Unlike OSQP, `qpth` is stateless, so no solver setup is needed here.
        We pre-construct the constant identity matrix `Q` for efficiency.

        Args:
            agent_count  : Number of agents in the flock
            barrier      : The ThermalBarrierFunction instance that defines
                           the safety boundary.
            cbf_alpha    : CBF constraint parameter α
            safety       : QP solver configuration model
            spatial_dims : Spatial dimensions (2D or 3D)
        """
        self.agent_count  = agent_count
        self.barrier      = barrier
        self.cbf_alpha    = cbf_alpha
        self.safety       = safety
        self.spatial_dims = spatial_dims
        self.Q            = torch.eye(self.spatial_dims, torch.float32, "cpu")

    def filter(
        self, 
        flock     : TensorDict, 
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
            flock     : The current observation data for the flock containing
                        temperature and temperature_grad tensors.
            u_nominal : The desired control action from the policy network.

        Returns:
            The batch of safe control actions `u*`.
        """
        h_values, h_grads = self.barrier.evaluate(flock)
        agent_count       = self.agent_count
        device            = u_nominal.device

        # Instantiate the QP solver with its configuration.
        solver = QPFunction(
            eps     = self.safety.qp_eps,
            maxIter = self.safety.qp_max_iter,
        )

        try:
            u_safe = solver(
                Q = self.Q.to(device).expand(agent_count, -1, -1),
                p = -u_nominal,
                G = -h_grads.unsqueeze(1),
                h = (self.cbf_alpha * h_values).unsqueeze(1),
                A = torch.empty(0, device=device),
                b = torch.empty(0, device=device),
            )
            
            # Check which agents had CBF active (u_safe != u_nominal)
            is_active = torch.norm(
                input = u_safe - u_nominal, 
                dim   = 1
            ) > self.barrier.activation_tolerance
            self.barrier.log_activation(is_active)

        except Exception as e:
            if self.safety.qp_on_failure == "nominal":
                return u_nominal
            raise ValueError(f"QP safety filter failed to find a solution: {e}")

        return u_safe.view_as(u_nominal)


class ThermalBarrierFunction:
    """
    Implements the temperature-based Control Barrier Function (CBF).
    
    The barrier function is defined as h(𝐬) = T_max - T(𝐬), which creates a
    safety boundary at T = T_max. This barrier ensures that agents remain within
    the safe set C = {𝐬 | h(𝐬) ≥ 0}, which corresponds to temperatures below
    the maximum survivable threshold.
    
    To enforce the invariance of this safe set, the Lie derivative condition 
    must be satisfied: 
    
        ∇h(𝐬) · u ≥ -α·h(𝐬), where α is a class-K function.
    """
    
    def __init__(
        self,
        activation_tolerance : float,
        max_temperature      : float
    ):
        """
        Initializes the thermal barrier function.
        
        Args:
            activation_tolerance : Threshold for determining CBF activation
            max_temperature      : Maximum safe temperature T_max for agents
        """
        self.activation_count     = 0
        self.activation_tolerance = activation_tolerance
        self.max_temperature      = max_temperature
        self.total_queries        = 0
    
    def evaluate(self, flock: TensorDict) -> tuple[Tensor, Tensor]:
        """
        Computes the barrier function h(𝐬) and its gradient ∇h(𝐬).
        
        The barrier function is h(𝐬) = T_max - T(𝐬), creating a boundary at
        T = T_max. The gradient ∇h(𝐬) = -∇T(𝐬) points away from high temperature
        regions, creating a "force" that pushes agents toward safety.
        
        Args:
            flock: The current observation data for the flock containing
                   temperature and temperature_grad tensors.
                
        Returns:
            A tuple containing (h_values, h_grads).
        """
        temp_grad = flock.get("temperature_grad", None)       
        h_grads   = -temp_grad if temp_grad is not None else None
        h_values  = self.max_temperature - flock["temperature"]
        
        return h_values, h_grads
    
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
        
    def log_activation(self, is_active: Tensor):
        """
        Records barrier function activations for debugging and monitoring.
        
        Args:
            is_active: Boolean tensor indicating which agents had the CBF
                       actively modify their control input.
        """
        self.activation_count += is_active.sum().item()
        self.total_queries    += is_active.shape[0]
            