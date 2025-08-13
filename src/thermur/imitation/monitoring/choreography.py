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


class TemporalMetric(Metric):
    """
    Base class for metrics that track temporal evolution.

    Provides state management for metrics that require comparison
    between consecutive timesteps to measure dynamics.
    """
    count : Tensor
    sum   : Tensor

    def __init__(self):
        """
        Initialize temporal metric with state tracking.
        """
        super().__init__()
        self.add_state("count",      th.tensor(0),   "sum")
        self.add_state("sum",        th.tensor(0.0), "sum")
        self.add_state("last_value", th.tensor(0.0), "mean")
        self.temporal_state = None

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
    

    def reset(self):
        """
        Reset metric state including temporal tracking.
        """
        super().reset()
        self.temporal_state = None


class InformationPropagationMetric(TemporalMetric):
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
        self.target_min = metrics.info_propagation_min_speed
        self.target_max = metrics.info_propagation_max_speed
        self.time_step  = metrics.info_propagation_time_step
    
    def update(self, batch: TensorDictBase):
        """
        Measure empirical propagation speed from velocity changes.
        
        Computes velocity perturbations and detects information propagation
        events when significant changes occur at the flock's edge. Propagation
        speed v = Δr/Δt where Δr is the spatial extent and Δt is the timestep.

        Args:
            batch: TensorDict containing position, velocity, timestep, and
                   trajectory_id tensors
        """
        current_velocities = batch["velocity"]
        self.last_value.fill_(0.0)
        
        if self.temporal_state is not None:
            previous_velocities = self.temporal_state
            
            velocity_changes = (
                current_velocities - previous_velocities
            ).norm(dim=1)
            
            if (threshold := velocity_changes.mean() + velocity_changes.std()) > 0:
                significant_changes = velocity_changes > threshold
                
                if significant_changes.any():
                    center = batch["position"].mean(dim=0)
                    radii  = (batch["position"] - center).norm(dim=1)
                    
                    if (significant_changes & (radii > radii.mean())).any():
                        propagation_speed = (
                            (radii.max() - radii.min()) / self.time_step
                        )
                        self.last_value.copy_(propagation_speed)
                        self.sum   += propagation_speed
                        self.count += 1
        
        self.temporal_state = current_velocities.clone().detach()


class SusceptibilityMetric(TemporalMetric):
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
    
    def reset(self):
        """
        Reset the metric state.
        """
        super().reset()
        self.polarization_queue.clear()
    
    def update(self, batch: TensorDictBase):
        """
        Compute susceptibility from learned policy's velocities.
        
        Updates the polarization history for the current trajectory and
        computes susceptibility χ = N·Var[Φ] where N is the number of agents
        and Var[Φ] is the variance of polarization over the time window.
        
        Args:
            batch: TensorDict containing velocity and trajectory_id
        """
        
        polarization = th.nn.functional.normalize(
            dim   = 1,
            input = batch["velocity"]
        ).mean(dim=0).norm()
        
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
        self.last_value.copy_(susceptibility)
        self.sum   += susceptibility
        self.count += 1


class TopologicalFidelityMetric(TemporalMetric):
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
        self.k_neighbors = mmm.k_neighbors
        self.last_value.fill_(1.0)
    
    def reset(self):
        """
        Reset the metric state with fidelity default of 1.0.
        """
        super().reset()
        self.last_value.fill_(1.0)
    
    def update(self, batch: TensorDictBase):
        """
        Update metric with neighbor consistency measurement.
        
        Computes the fraction of k-nearest neighbors maintained from the
        previous timestep. Fidelity φ = |N_t ∩ N_{t-1}| / k where N_t is
        the set of k-nearest neighbors at timestep t.
        
        Args:
            batch: TensorDict containing position, timestep, and trajectory_id
        """
        
        _, indices = th.cdist(batch["position"], batch["position"]).topk(
            k       = self.k_neighbors + 1,
            largest = False
        )
        current_neighbors = indices[:, 1:]
        self.last_value.fill_(1.0)
        if self.temporal_state is not None:
            previous_neighbors = self.temporal_state
            
            overlap = (
                previous_neighbors.unsqueeze(2) == current_neighbors.unsqueeze(1)
            ).any(dim=2).sum(dim=1).float()
            
            fidelity = overlap.mean() / self.k_neighbors
            self.last_value.copy_(fidelity)
            self.sum   += fidelity
            self.count += 1
        
        self.temporal_state = current_neighbors.clone().detach()


class ChoreographyCollector(th.nn.Module):
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
        super().__init__()
        self.metrics = metrics
        self.mmm     = mmm
        
        self.choreography_metrics = MetricCollection({
            "info_propagation"     : InformationPropagationMetric(metrics),
            "susceptibility"       : SusceptibilityMetric(metrics, mmm),
            "topological_fidelity" : TopologicalFidelityMetric(mmm)
        })
        
        self.trajectory_count     = 0
        self.trajectory_frequency = metrics.trajectory_frequency
        self.timestep_history     = []
    
    def _log_timestep_metrics(self, pl_module: LightningModule | None):
        """
        Log timestep-by-timestep metrics to WandB.
        
        Logs temporal evolution of choreography metrics, enabling visualization
        of how information propagation, susceptibility, and topological fidelity
        change over the course of trajectories.
        
        Args:
            pl_module: Lightning module with WandB logger access
        """
        if not pl_module:
            return
            
        for timestep_metrics in self.timestep_history:
            metrics_dict = {
                f"timestep/{k}": v for k, v in timestep_metrics.items()
                if k != "timestep"
            }
            metrics_dict["timestep/global_step"] = timestep_metrics["timestep"]
            pl_module.log_dict(metrics_dict)
        
        self.timestep_history.clear()

    def _log_metrics(self, pl_module: LightningModule | None):
        """
        Log computed metrics to WandB.
        
        Sends choreography metrics to WandB under the "choreography/" namespace,
        allowing them to be tracked alongside training metrics in the same run.
        
        Args:
            pl_module: Lightning module with WandB logger access
        """
        if not pl_module:
            return
            
        computed = self.choreography_metrics.compute()
        pl_module.log_dict({
            f"choreography/{k}": v 
            for k, v in computed.items()
        })
        self.choreography_metrics.reset()
    
    def analyze_trajectory(
        self,
        trajectory : TensorDictBase,
        pl_module  : LightningModule | None = None
    ) -> list[dict]:
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
        
        Returns:
            List of per-timestep metrics
        """
        num_timesteps = trajectory["position"].shape[0]
        base_timestep = self.trajectory_count * num_timesteps
        prev_values   = {"susceptibility" : 0.0, "topological_fidelity" : 1.0}
        timestep_data = []
        
        for t in range(num_timesteps):
            frame = TensorDict({
                k: v[t] for k, v in trajectory.items()
            }, batch_size=[])
            
            for metric in self.choreography_metrics.values():
                metric.update(frame)
            
            current_values = {
                name: metric.last_value.item()
                for name, metric in self.choreography_metrics.items()
            }
            current_values["timestep"] = base_timestep + t
            
            if t > 0:
                for name in ["susceptibility", "topological_fidelity"]:
                    current_values[f"delta_{name}"] = (
                        current_values[name] - prev_values.get(name, 0)
                    )
            
            timestep_data.append(current_values)
            self.timestep_history.append(current_values)
            prev_values = current_values
        
        self.trajectory_count += 1
        
        if self.trajectory_count % self.trajectory_frequency == 0:
            self._log_metrics(pl_module)
            self._log_timestep_metrics(pl_module)
        
        return timestep_data
    
    def compute_temporal_changes(
        self,
        trajectories : list[TensorDictBase],
        pl_module    : LightningModule | None = None
    ) -> dict[int, list[dict]]:
        """
        Compute and log temporal changes in choreography metrics.
        
        Processes trajectories to extract per-timestep metrics and their
        temporal derivatives, providing insight into how the flock's
        dynamics evolve over time.
        
        Args:
            trajectories : List of TensorDicts, each with shape [T, n_agents, features]
            pl_module    : Optional Lightning module for WandB logging
        
        Returns:
            Dictionary mapping trajectory index to list of timestep metrics
        """
        return {
            idx: timestep_data
            for idx, trajectory in enumerate(trajectories)
            if (
                self.choreography_metrics.reset(),
                timestep_data := self.analyze_trajectory(trajectory, pl_module)
            )[1]
        }
    
    def get_summary(self) -> dict[str, Tensor]:
        """
        Get final summary of choreography metrics.
        
        Returns:
            Dictionary of computed metric values
        """
        return self.choreography_metrics.compute()