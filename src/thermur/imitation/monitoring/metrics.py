"""
Unified metrics collection for imitation learning training and evaluation.

This module provides a centralized MetricsCollector that manages all metrics
for the training pipeline, including imitation learning losses, core evaluation
metrics, and runtime performance tracking. The collector integrates seamlessly
with PyTorch Lightning's logging system and Weights & Biases.
"""
from config.imitation.schemas.monitoring import MonitoringModel
from pytorch_lightning                   import LightningModule
from tensordict                          import TensorDict
from torch                               import Tensor
from torchmetrics                        import MeanAbsoluteError, MeanSquaredError
from torchmetrics                        import MetricCollection, R2Score, Metric
from torchmetrics.image                  import StructuralSimilarityIndexMeasure
from typing                              import Optional

import torch


class CohesionMetric(Metric):
    """
    Measures graph connectivity via the second smallest eigenvalue (λ₂).
    
    The algebraic connectivity (Fiedler value) quantifies how well-connected
    the communication graph is, with higher values indicating stronger cohesion.
    A disconnected graph has λ₂ = 0, while a complete graph has maximum λ₂.
    """
    
    def __init__(self):
        """
        Sets up state variables for tracking the sum of λ₂ values and count
        of measurements for computing the running average.
        """
        super().__init__()
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("lambda2_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
    
    def compute(self) -> Tensor:
        """
        Compute the average algebraic connectivity.
        
        Returns:
            Average λ₂ (Fiedler value) across all graph updates
        """
        return (
            self.lambda2_sum / self.count 
                if self.count > 0 
                else torch.tensor(0.0)
        )
    
    def update(
        self,
        edge_index : Tensor,
        num_agents : int
    ):
        """
        Computes the second smallest eigenvalue of the graph Laplacian matrix:
        L = D - A, where D is the degree matrix and A is the adjacency matrix.
        
        Args:
            edge_index : Graph connectivity [2, num_edges] in COO format
            num_agents : Total number of agents in the graph
        """
        if edge_index.numel() == 0 or num_agents < 2:
            self.lambda2_sum += 0.0
            self.count += 1
            return
        
        device = edge_index.device
        
        adj_matrix = torch.zeros((num_agents, num_agents), device=device)
        adj_matrix.index_put_(
            (edge_index[0], edge_index[1]), 
            torch.ones(edge_index.shape[1], device=device)
        )
        adj_matrix = ((adj_matrix + adj_matrix.T) > 0).float()
        
        laplacian = torch.diag(adj_matrix.sum(dim=1)) - adj_matrix
        
        try:
            eigenvalues = torch.linalg.eigvalsh(laplacian)
            lambda2 = (eigenvalues[1] 
                      if eigenvalues.numel() > 1 
                      else torch.tensor(0.0, device=device))
        except:
            lambda2 = torch.tensor(0.0, device=device)
            
        self.lambda2_sum += lambda2
        self.count += 1


class ColorAccuracyMetric(Metric):
    """
    Measures the Mean Absolute Error (mae_color) between sensed and displayed temperatures.
    
    This metric quantifies how accurately the swarm can display temperature
    information through their RGB LEDs, using a heat colormap mapping where
    blue represents cold and red represents hot temperatures.
    """
    
    def __init__(self, monitoring: MonitoringModel):
        """
        Initialize the color accuracy metric.
        
        Args:
            monitoring : Monitoring configuration model
        """
        super().__init__()
        self.temp_max = monitoring.color_temp_max
        self.temp_min = monitoring.color_temp_min
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("error_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
    
    def _rgb_to_temperature(self, rgb: Tensor) -> Tensor:
        """
        Convert RGB color to temperature using red channel as indicator.
        
        Args:
            rgb : RGB colors [N, 3] in range [0, 1]
            
        Returns:
            Reconstructed temperatures [N] in Kelvin
        """
        return torch.lerp(self.temp_min, self.temp_max, rgb[..., 0])
    
    def _temperature_to_rgb(self, temperature: Tensor) -> Tensor:
        """
        Map temperature to RGB color using heat colormap.
        
        Blue (cold) -> Green (medium) -> Red (hot)
        
        Args:
            temperature : Temperature values [N] in Kelvin
            
        Returns:
            RGB colors [N, 3] in range [0, 1]
        """
        temp_norm = torch.clamp(
            (temperature - self.temp_min) / (self.temp_max - self.temp_min),
            0, 1
        )
        
        return torch.stack([
            torch.clamp(2 * temp_norm - 0.5, 0, 1),
            torch.where(
                temp_norm < 0.5,
                2 * temp_norm,
                2 * (1 - temp_norm)
            ),
            torch.clamp(1 - 2 * temp_norm, 0, 1)
        ], dim=-1)
    
    def compute(self) -> Tensor:
        """
        Compute the mean absolute error in temperature display.
        
        Returns:
            MAE between sensed and reconstructed temperatures in Kelvin
        """
        return (
            self.error_sum / self.count 
                if self.count > 0 
                else torch.tensor(0.0)
        )
    
    def update(
        self,
        sensed_temperature : Tensor,
        displayed_rgb      : Optional[Tensor] = None
    ):
        """
        Update metric with temperature and color data.
        
        Args:
            sensed_temperature : Actual temperatures sensed by agents [N]
            displayed_rgb      : RGB colors displayed by agents [N, 3] or None
        """
        sensed_temperature = sensed_temperature.flatten()
            
        displayed_rgb = displayed_rgb or self._temperature_to_rgb(sensed_temperature)
        error = (sensed_temperature - self._rgb_to_temperature(displayed_rgb)).abs()
        
        self.error_sum += error.sum()
        self.count     += sensed_temperature.numel()


class EnergyConsumptionMetric(Metric):
    """
    Estimates average power consumption based on control inputs.
    
    Uses a simplified quadrotor power model:
    P ∝ ||u_safe - g||^k
    
    where:
        - u_safe : safety-filtered control vector
        - g : gravity vector pointing downward
        - k : power exponent (typically 1.5 for quadrotors)
    """
    
    def __init__(
        self,
        gravity    : float,
        monitoring : MonitoringModel
    ):
        """
        Initialize the energy metric.
        
        Args:
            gravity    : Gravitational acceleration from physics config
            monitoring : Monitoring configuration model
        """
        super().__init__()
        self.gravity        = gravity
        self.power_exponent = monitoring.power_exponent
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("power_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
    
    def compute(self) -> Tensor:
        """
        Compute the average power consumption per agent.
        
        Returns:
            Average power consumption in arbitrary units
        """
        return (
            self.power_sum / self.count 
                if self.count > 0 
                else torch.tensor(0.0)
        )
    
    def update(self, u_safe: Tensor):
        """
        Computes instantaneous power from thrust vector magnitude.
        
        Args:
            u_safe : Safety-filtered control actions [N, 3] (m/s²)
        """
        gravity_vec = torch.zeros_like(u_safe)
        gravity_vec[..., 2] = -self.gravity
        
        power = (u_safe - gravity_vec).norm(dim=-1).pow(self.power_exponent)
        
        self.power_sum += power.sum()
        self.count     += u_safe.shape[0]


class LegibilitySSIMMetric(Metric):
    """
    Computes Structural Similarity Index Measure (ssim) between swarm and wind fields.
    
    This metric measures how well the swarm's collective motion pattern matches
    the underlying wind field, quantifying the visual "legibility" of the display.
    Uses kernel density estimation to render velocity fields onto 2D grids.
    """
    
    def __init__(
        self,
        bounds_max : list[float],
        monitoring : MonitoringModel
    ):
        """
        Initialize the legibility metric.
        
        Args:
            bounds_max : Maximum workspace bounds from physics config
            monitoring : Monitoring configuration model
        """
        super().__init__()
        self.bounds_max  = bounds_max
        self.grid_size   = monitoring.legibility_grid_size
        self.kernel_size = monitoring.legibility_kernel_size
        self.sigma       = monitoring.legibility_sigma
        
        self.ssim_metric = StructuralSimilarityIndexMeasure(
            data_range=1.0,
            kernel_size=self.kernel_size,
            reduction='elementwise_mean'
        )
        
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("ssim_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
    
    def _render_velocity_field(
        self,
        bounds_max : Tensor,
        bounds_min : Tensor,
        positions  : Tensor,
        velocities : Tensor
    ) -> Tensor:
        """
        Projects 3D positions and velocities onto the x-y plane and applies
        Gaussian kernel density estimation to create a smooth velocity field.
        
        Args:
            bounds_max : Maximum workspace bounds [3]
            bounds_min : Minimum workspace bounds [3]
            positions  : Agent positions [N, 3]
            velocities : Agent velocities [N, 3]
            
        Returns:
            2D tensor representing the velocity magnitude field
        """
        device = positions.device
        
        pos_2d = positions[:, :2]
        vel_magnitude = velocities[:, :2].norm(dim=1)
        
        grid_pos = ((pos_2d - bounds_min[:2]) / 
                    (bounds_max[:2] - bounds_min[:2]) * 
                    (self.grid_size - 1))
        
        field = torch.zeros((self.grid_size, self.grid_size), device=device)
        
        coords = torch.stack(torch.meshgrid(
            torch.arange(self.grid_size, device=device),
            torch.arange(self.grid_size, device=device),
            indexing='xy'
        ), dim=-1).float()
        
        grid_pos_expanded = grid_pos.unsqueeze(0).unsqueeze(0)
        coords_expanded = coords.unsqueeze(2)
        
        dist_sq = ((coords_expanded - grid_pos_expanded) ** 2).sum(dim=-1)
        weights = torch.exp(-dist_sq / (2 * self.sigma**2))
        field = (weights * vel_magnitude.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        
        field_max = field.max()
        return field / field_max if field_max > 0 else field
    
    def compute(self) -> Tensor:
        """
        Compute the average SSIM score.
        
        Returns:
            Average SSIM score in range [0, 1], where 1 indicates perfect match
        """
        return (self.ssim_sum / self.count 
                if self.count > 0 
                else torch.tensor(0.0))
    
    def update(
        self,
        bounds_max : Tensor,
        bounds_min : Tensor,
        positions  : Tensor,
        velocities : Tensor,
        wind_field : Tensor
    ):
        """
        Computes a simplified SSIM using luminance and contrast terms:
        
        SSIM = (2*μ_x*μ_y + C1)(2*σ_xy + C2) / 
               ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))
        
        where:
            - μ_x, μ_y : mean values of swarm and wind fields
            - σ_x, σ_y : standard deviations of fields
            - σ_xy : covariance between fields
            - C1, C2 : small constants to avoid division by zero
        
        Args:
            bounds_max : Maximum workspace bounds [3]
            bounds_min : Minimum workspace bounds [3]
            positions  : Agent positions [N, 3]
            velocities : Agent velocities [N, 3]
            wind_field : Ground truth wind velocities [N, 3]
        """
        swarm_field = self._render_velocity_field(
            bounds_min, bounds_max, positions, velocities
        ).unsqueeze(0).unsqueeze(0)
        
        wind_field_rendered = self._render_velocity_field(
            bounds_min, bounds_max, positions, wind_field
        ).unsqueeze(0).unsqueeze(0)
        
        ssim_value = self.ssim_metric(swarm_field, wind_field_rendered)
        
        self.ssim_sum += ssim_value
        self.count    += 1


class MetricsCollector:
    """
    Centralized metric collection and management for training and evaluation.
    
    This class owns all TorchMetrics instances and provides methods to update
    and log metrics throughout the training process. It handles:
    
    1. Imitation learning metrics (MSE, RMSE, MAE, R²)
    2. Core evaluation metrics (thermal safety, legibility, cohesion, energy, color)
    3. Runtime metrics (CBF activations, trajectories)
    4. Event logging for debugging
    
    The collector is designed to work with PyTorch Lightning's logging system
    and automatically syncs metrics to Weights & Biases through the configured
    logger.
    """
    
    def __init__(
        self,
        bounds_max      : list[float],
        gravity         : float,
        max_temperature : float,
        monitoring      : MonitoringModel,
        output_dim      : int
    ):
        """
        Initialize the metrics collector with all metric instances.
        
        Args:
            bounds_max      : Maximum workspace bounds from physics config
            gravity         : Gravitational acceleration from physics config
            max_temperature : Maximum safe temperature from flock config
            monitoring      : Monitoring configuration model
            output_dim      : Dimension of the policy output
        """
        self.bounds_max      = bounds_max
        self.gravity         = gravity
        self.max_temperature = max_temperature
        self.monitoring      = monitoring
        self.output_dim      = output_dim
        
        self._init_imitation_metrics(output_dim=output_dim)
        self._init_evaluation_metrics()
        self._init_runtime_trackers()
    
    def _init_evaluation_metrics(self):
        """
        Creates TorchMetrics instances for all five core performance metrics.
        """
        self.train_evaluation = MetricCollection({
            "avg_power"  : EnergyConsumptionMetric(self.gravity, self.monitoring),
            "λ₂"         : CohesionMetric(),
            "mae_color"  : ColorAccuracyMetric(self.monitoring),
            "ssim"       : LegibilitySSIMMetric(self.bounds_max, self.monitoring),
            "tvr"        : ThermalSafetyMetric(self.max_temperature),
        })
        self.val_evaluation = self.train_evaluation.clone(prefix="val_")
    
    def _init_imitation_metrics(self, output_dim: int):
        """
        Creates standard regression metrics for behavioral cloning loss.
        
        Args:
            output_dim : Dimensionality of action space for R² calculation
        """
        self.train_imitation = MetricCollection({
            "mae"  : MeanAbsoluteError(),
            "mse"  : MeanSquaredError(),
            "r2"   : R2Score(num_outputs=output_dim),
            "rmse" : MeanSquaredError(False),
        })
        self.val_imitation = self.train_imitation.clone(prefix="val_")
    
    def _init_runtime_trackers(self):
        """
        Sets up counters for tracking CBF activations and other runtime events
        that don't fit into the standard TorchMetrics framework.
        """
        self.cbf_activation_count = 0
        self.total_steps          = 0
    
    def get_cbf_activation_rate(self) -> float:
        """
        Calculate the rate of CBF activations during the current epoch.
        
        Returns:
            CBF activation rate as a float between 0 and 1
        """
        return (
            self.cbf_activation_count / self.total_steps 
                if self.total_steps > 0 
                else 0.0
        )
    
    def log_all_metrics(
        self,
        module      : LightningModule,
        phase       : str,
        loss        : Optional[Tensor] = None,
        predictions : Optional[Tensor] = None,
        targets     : Optional[Tensor] = None
    ):
        """
        Handles logging of all metric types and ensures proper
        synchronization with the Lightning logger (including W&B).
        
        Args:
            module      : Lightning module for logging
            phase       : Training phase ("train" or "val")
            loss        : Optional loss value to log
            predictions : Optional model predictions for dimension-wise logging
            targets     : Optional targets for dimension-wise logging
        """
        is_train = phase == "train"
        
        def log_metric(name, value, on_step=None, prog_bar=False):
            """
            Helper to log metrics with consistent settings.
            
            Args:
                name     : Metric name with phase prefix (e.g., "train/loss")
                value    : Scalar metric value to log
                on_step  : Whether to log per step (defaults to is_train)
                prog_bar : Whether to show in progress bar
            """
            module.log(
                name     = name,
                value    = value,
                on_epoch = True,
                on_step  = is_train if on_step is None else on_step,
                prog_bar = prog_bar
            )
        
        if loss is not None:
            log_metric(
                name     = f"{phase}/loss",
                prog_bar = True,
                value    = loss
            )
        
        for metrics, on_step in [
            (self.train_imitation  if is_train else self.val_imitation,  is_train),
            (self.train_evaluation if is_train else self.val_evaluation, False)
        ]:
            module.log_dict(
                dictionary = metrics,
                on_epoch   = True,
                on_step    = on_step
            )
        
        if predictions is not None and targets is not None:
            for i, dim in enumerate(["x", "y", "z"][:self.output_dim]):
                log_metric(
                    name    = f"{phase}/velocity_{dim}_mse",
                    on_step = False,
                    value   = (predictions[..., i] - targets[..., i]).pow(2).mean()
                )
        
        if is_train:
            log_metric(
                name    = "train/cbf_activation_rate",
                on_step = False,
                value   = self.get_cbf_activation_rate()
            )
    
    def log_cbf_activation(self, batch: TensorDict):
        """
        Updates internal counters used to compute CBF activation rate.
        
        Args:
            batch : TensorDict containing CBF activation information
        """
        if "cbf_active" in batch:
            self.cbf_activation_count += batch["cbf_active"].sum().item()
            self.total_steps          += batch["cbf_active"].numel()
    
    def reset_runtime_metrics(self):
        """
        Should be called at epoch boundaries to ensure accurate per-epoch
        CBF activation rates.
        """
        self.cbf_activation_count = 0
        self.total_steps          = 0
    
    def update_evaluation_metrics(
        self,
        batch : TensorDict,
        phase : str
    ):
        """
        Extract relevant fields from batch and update each metric.
        
        Handles missing fields gracefully.
        
        Args:
            batch : TensorDict containing simulation state and actions
            phase : Training phase ("train" or "val")
        """
        metrics = (
            self.train_evaluation if phase == "train" else self.val_evaluation
        )
        
        if "temperature" in batch:
            metrics["tvr"].update(temperature=batch["temperature"])
            metrics["mae_color"].update(batch["temperature"])
        
        if all(k in batch for k in ["position", "velocity", "wind"]):
            device = batch["position"].device
            metrics["ssim"].update(
                bounds_max = torch.tensor(self.bounds_max, device=device),
                bounds_min = torch.zeros(3, device=device),
                positions  = batch["position"],
                velocities = batch["velocity"],
                wind_field = batch["wind"]
            )
        
        if "edge_index" in batch:
            metrics["λ₂"].update(
                edge_index=batch["edge_index"], 
                num_agents=batch.get("position", torch.empty(0)).shape[0]
            )
        
        if u_control := batch.get("u_safe") or batch.get("action"):
            metrics["avg_power"].update(u_safe=u_control)
    
    def update_imitation_metrics(
        self,
        phase       : str,
        predictions : Tensor,
        targets     : Tensor
    ):
        """
        Update imitation learning metrics with predictions and targets.
        
        Args:
            phase       : Training phase ("train" or "val")
            predictions : Model predictions [batch_size, output_dim]
            targets     : Expert actions [batch_size, output_dim]
        """
        imitation_metrics = (
            self.train_imitation if phase == "train" else self.val_imitation
        )
        imitation_metrics.update(predictions, targets)


class ThermalSafetyMetric(Metric):
    """
    Tracks the thermal violation rate (TVR): P(T_agent > T_max).
    
    This metric monitors how often agents exceed the maximum safe temperature
    threshold, which is critical for mission success and agent survival.
    Violations indicate failures of the safety system.
    """
    
    def __init__(self, max_temperature: float):
        """
        Initialize the thermal safety metric.
        
        Args:
            max_temperature : Maximum safe temperature from flock config
        """
        super().__init__()
        self.max_temperature = max_temperature
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("violations", default=torch.tensor(0.0), dist_reduce_fx="sum")
    
    def compute(self) -> Tensor:
        """
        Compute the violation rate.
        
        Returns:
            Fraction of temperature readings that exceeded T_max
        """
        return (
            self.violations / self.total 
                if self.total > 0 
                else torch.tensor(0.0)
        )
    
    def update(self, temperature: Tensor):
        """
        Update metric with new temperature readings.
        
        Args:
            temperature : Agent temperatures [N] or [N, 1] in Kelvin
        """
        temperature = temperature.flatten()
            
        self.violations += (
            (temperature > self.max_temperature).sum(dtype=torch.float32)
        )
        self.total      += temperature.numel()