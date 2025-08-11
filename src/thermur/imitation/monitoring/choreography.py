"""
Choreography analysis for temporal murmuration dynamics.

This module analyzes the expert controller's temporal murmuration properties
during trajectory collection. It contains metrics that require sequential data,
such as information propagation speed, susceptibility variance, and topological
fidelity across timesteps.
"""
from __future__   import annotations
from collections  import deque
from tensordict   import TensorDict, TensorDictBase
from torchmetrics import Metric, MetricCollection
from typing       import TYPE_CHECKING

if TYPE_CHECKING:
    from config.imitation.controller import MurmurationModel
    from config.imitation.monitoring import MetricsModel
    from pytorch_lightning           import LightningModule
    from torch                       import Tensor

import torch as th
import wandb


class AveragingMetric(Metric):
    """
    Base class for metrics that compute running averages.

    Provides common sum/count state management and averaging logic
    for metrics that accumulate values over batches. Subclasses should
    implement the update() method to add values to the sum.
    """
    count : Tensor
    sum   : Tensor

    def __init__(self):
        """
        Initialize state variables for computing running averages.

        Creates two state variables that are synchronized across distributed
        training: 'count' for tracking the number of measurements and 'sum'
        for accumulating the values to be averaged.
        """
        super().__init__()
        self.add_state("count", default=th.tensor(0),   dist_reduce_fx="sum")
        self.add_state("sum",   default=th.tensor(0.0), dist_reduce_fx="sum")

    def compute(self) -> Tensor:
        """
        Compute the average of accumulated values.

        Returns the mean of all values accumulated via the update method,
        or zero if no values have been recorded yet.
        """
        return (
            self.sum / self.count if self.count > 0 
            else th.zeros_like(self.sum)
        )


class InformationPropagationMetric(AveragingMetric):
    """
    Measures empirical information propagation speed through the learned flock.

    Tracks how velocity perturbations propagate from the flock's edge to center,
    comparing against the theoretical range v_info ∈ [15, 45] m/s observed in
    starling murmurations (Cavagna et al. 2010). This empirical measurement
    reveals whether the learned policy maintains proper information transfer.
    """
    
    def __init__(self, metrics: MetricsModel):
        """
        Initialize with target propagation speed range.

        Args:
            metrics : Metrics configuration containing propagation speeds
        """
        super().__init__()
        self.previous_velocities = None
        self.target_min          = metrics.info_propagation_min_speed
        self.target_max          = metrics.info_propagation_max_speed
        self.time_step           = metrics.info_propagation_time_step
    
    def update(self, batch: TensorDictBase):
        """
        Measure empirical propagation speed from velocity changes.

        Args:
            batch: TensorDict containing position and velocity
        """
        if self.previous_velocities is None:
            self.previous_velocities = batch["velocity"].clone()
            return

        v_changes = (batch["velocity"] - self.previous_velocities).norm(dim=1)
        
        if (threshold := v_changes.mean() + v_changes.std()) > 0 and \
           (significant := v_changes > threshold).any():
            center = batch["position"].mean(dim=0)
            radii  = (batch["position"] - center).norm(dim=1)
            
            if (significant & (radii > radii.mean())).any():
                propagation_speed = (
                    (radii.max() - radii.min()) / self.time_step
                )
                
                self.sum   += propagation_speed
                self.count += 1
        
        self.previous_velocities = batch["velocity"].clone()


class SusceptibilityMetric(AveragingMetric):
    """
    Computes learned flock susceptibility χ = N · Var[Φ] to assess critical dynamics.
    
    Measures how well the learned policy maintains critical state susceptibility
    compared to the expert controller. From Cavagna et al. (2010), susceptibility
    quantifies the flock's responsiveness to perturbations:
    
        χ = N · ⟨(Φ - ⟨Φ⟩)²⟩
    
    where Φ = |Σ_i 𝐬_i|/N is the polarization order parameter. At critical
    state (χ ∈ [5, 20]), information propagates near-instantaneously.
    """
    
    def __init__(self, metrics: MetricsModel, mmm: MurmurationModel):
        """
        Initialize with target susceptibility range and polarization tracking.
        
        Args:
            metrics : Metrics configuration containing susceptibility range
            mmm     : Murmuration model with polarization window parameter
        """
        super().__init__()
        self.polarization_queue = deque(maxlen=mmm.polarization_window)
        self.target_min         = metrics.susceptibility_min
        self.target_max         = metrics.susceptibility_max
    
    def update(self, batch: TensorDictBase):
        """
        Compute susceptibility from learned policy's velocities.
        
        Args:
            batch: TensorDict containing velocity
        """
        spin_vectors = (
            batch["velocity"] / 
            batch["velocity"].norm(dim=1, keepdim=True).clamp_min(1e-8)
        )
        
        polarization = spin_vectors.mean(dim=0).norm()
        self.polarization_queue.append(polarization.item())
        
        variance = (
            th.tensor(
                data   = list(self.polarization_queue), 
                device = batch["velocity"].device,
                dtype  = batch["velocity"].dtype
            ).var()
            if len(self.polarization_queue) > 1
            else polarization * (1 - polarization)
        )
        
        susceptibility = batch["velocity"].shape[0] * variance
        
        self.sum   += susceptibility
        self.count += 1


class TopologicalFidelityMetric(AveragingMetric):
    """
    Measures consistency of k-nearest neighbor connections.
    
    Tracks how well agents maintain their topological neighborhoods
    over time, essential for information flow in murmurations.
    """
    
    def __init__(self, mmm: MurmurationModel):
        """
        Initialize with number of topological neighbors.
        
        Args:
            mmm: Murmuration model containing k_neighbors
        """
        super().__init__()
        self.k_neighbors        = mmm.k_neighbors
        self.previous_neighbors = None
    
    def update(self, batch: TensorDictBase):
        """
        Update metric with neighbor consistency measurement.
        
        Args:
            batch: TensorDict containing position
        """
            
        _, indices = th.cdist(batch["position"], batch["position"]).topk(
            k       = self.k_neighbors + 1, 
            largest = False
        )

        current_neighbors = indices[:, 1:]
        if self.previous_neighbors is not None:
            overlap_count = (
                self.previous_neighbors.unsqueeze(2) == 
                current_neighbors.unsqueeze(1)
            ).any(dim=2).sum(dim=1).float()
            
            self.sum   += overlap_count.mean() / self.k_neighbors
            self.count += 1
        
        self.previous_neighbors = current_neighbors.clone()


class ChoreographyCollector:
    """
    Centralized choreography collection for temporal murmuration dynamics.
    
    This collector manages temporal metrics that require sequential data
    continuity during expert trajectory collection. It validates that the
    expert controller exhibits the expected biological murmuration properties
    over time, tracking:
    
    1. Information propagation speed (15-45 m/s from Cavagna et al. 2010)
    2. Critical state susceptibility (χ ∈ [5, 20])
    3. Topological neighbor consistency

    The collector processes complete trajectories with temporal continuity,
    computing metrics that cannot be evaluated on random batches.
    """
    
    def __init__(
        self,
        metrics : MetricsModel,
        mmm     : MurmurationModel
    ):
        """
        Initialize the choreography collector with temporal metrics.
        
        Creates metric instances for tracking information propagation,
        susceptibility variance, and topological consistency.
        
        Args:
            metrics : Metrics configuration with target ranges
            mmm     : Murmuration dynamics configuration
        """
        self.metrics = metrics
        self.mmm     = mmm
        
        self.choreography_metrics = MetricCollection({
            "info_propagation"     : InformationPropagationMetric(metrics),
            "susceptibility"       : SusceptibilityMetric(metrics, mmm),
            "topological_fidelity" : TopologicalFidelityMetric(mmm)
        })
        
        self.trajectory_count     = 0
        self.trajectory_frequency = metrics.trajectory_frequency
    
    def analyze_trajectory(
        self,
        trajectory : TensorDictBase,
        pl_module  : LightningModule | None = None
    ):
        """
        Analyze a complete trajectory for temporal murmuration properties.
        
        Processes a sequence of frames with temporal continuity, computing
        metrics that track changes and patterns over time. Each frame is
        evaluated sequentially to maintain temporal relationships required
        for metrics like topological fidelity and information propagation.
        
        Args:
            trajectory : TensorDict with shape [T, n_agents, features] where T is
                        the number of sequential timesteps
            pl_module  : Optional Lightning module for accessing the WandB logger
        """
        num_timesteps = trajectory["position"].shape[0]
        
        for t in range(num_timesteps):
            frame = TensorDict({
                k: v[t] for k, v in trajectory.items()
            }, batch_size=[])
            
            for metric in self.choreography_metrics.values():
                metric.update(frame)
        
        self.trajectory_count += 1
        
        if self.trajectory_count % self.trajectory_frequency == 0:
            self._log_metrics(pl_module)
    
    def _log_metrics(self, pl_module: LightningModule | None):
        """
        Log computed metrics to WandB.
        
        Sends choreography metrics to WandB under the "choreography/" namespace,
        allowing them to be tracked alongside training metrics in the same run.
        
        Args:
            pl_module: Lightning module with WandB logger access
        """
        computed = self.choreography_metrics.compute()
        
        if wandb.run is not None:
            wandb.log({
                f"choreography/{k}": v 
                for k, v in computed.items()
            }, step = self.trajectory_count)
            
        elif pl_module and hasattr(pl_module, "logger") and pl_module.logger:
            pl_module.logger.log_metrics({
                f"choreography/{k}": v 
                for k, v in computed.items()
            }, step = self.trajectory_count)
        
        self.choreography_metrics.reset()
    
    def get_summary(self) -> dict[str, Tensor]:
        """
        Get final summary of choreography metrics.
        
        Returns:
            Dictionary of computed metric values
        """
        return self.choreography_metrics.compute()