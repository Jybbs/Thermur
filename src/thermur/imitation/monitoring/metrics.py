"""
Unified metrics collection for imitation learning training and evaluation.

This module provides a centralized MetricsCollector that manages all metrics
for the training pipeline, including imitation learning losses, core evaluation
metrics, and runtime performance tracking. The collector integrates seamlessly
with PyTorch Lightning's logging system and Weights & Biases.
"""
from pytorch_lightning   import LightningModule
from tensordict          import TensorDict
from torch               import Tensor
from torchmetrics        import MeanAbsoluteError, MeanSquaredError
from torchmetrics        import MetricCollection, R2Score, Metric
from torchmetrics.image  import StructuralSimilarityIndexMeasure
from typing              import Optional

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
        return torch.where(self.count > 0, self.lambda2_sum / self.count, torch.tensor(0.0))
    
    def update(self, edge_index: Tensor, num_agents: int):
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
        adj_matrix[edge_index[0], edge_index[1]] = 1.0
        
        adj_matrix = adj_matrix + adj_matrix.T
        adj_matrix = (adj_matrix > 0).float()
        
        degrees = adj_matrix.sum(dim=1)
        degree_matrix = torch.diag(degrees)
        
        laplacian = degree_matrix - adj_matrix
        
        try:
            eigenvalues = torch.linalg.eigvalsh(laplacian)
            lambda2 = eigenvalues[1] if eigenvalues.numel() > 1 else torch.tensor(0.0, device=device)
        except:
            lambda2 = torch.tensor(0.0, device=device)
            
        self.lambda2_sum += lambda2
        self.count += 1


class ColorAccuracyMetric(Metric):
    """
    Measures the Mean Absolute Error between sensed and displayed temperatures.
    
    This metric quantifies how accurately the swarm can display temperature
    information through their RGB LEDs, using a heat colormap mapping where
    blue represents cold and red represents hot temperatures.
    """
    
    def __init__(self, temp_max: float = 473.0, temp_min: float = 273.0):
        """
        Initialize the color accuracy metric.
        
        Args:
            temp_max : Maximum temperature for color mapping (Kelvin)
            temp_min : Minimum temperature for color mapping (Kelvin)
        """
        super().__init__()
        self.temp_max = temp_max
        self.temp_min = temp_min
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("error_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
    
    def _rgb_to_temperature(self, rgb: Tensor) -> Tensor:
        """
        Uses a simplified inverse based on the red channel as primary
        temperature indicator. In practice, this would require calibration.
        
        Args:
            rgb : RGB colors [N, 3] in range [0, 1]
            
        Returns:
            Reconstructed temperatures [N] in Kelvin
        """
        temp_norm = rgb[..., 0]
        return self.temp_min + temp_norm * (self.temp_max - self.temp_min)
    
    def _temperature_to_rgb(self, temperature: Tensor) -> Tensor:
        """
        Maps temperature to color using a smooth gradient:
        - Blue (cold): T = T_min
        - Green (medium): T = (T_min + T_max) / 2  
        - Red (hot): T = T_max
        
        Args:
            temperature : Temperature values [N] in Kelvin
            
        Returns:
            RGB colors [N, 3] in range [0, 1]
        """
        temp_norm = (temperature - self.temp_min) / (self.temp_max - self.temp_min)
        temp_norm = torch.clamp(temp_norm, 0, 1)
        
        r = torch.clamp(2 * temp_norm - 0.5, 0, 1)
        g = torch.where(
            temp_norm < 0.5,
            2 * temp_norm,
            2 * (1 - temp_norm)
        )
        b = torch.clamp(1 - 2 * temp_norm, 0, 1)
        
        return torch.stack([r, g, b], dim=-1)
    
    def compute(self) -> Tensor:
        """
        Compute the mean absolute error in temperature display.
        
        Returns:
            MAE between sensed and reconstructed temperatures in Kelvin
        """
        return torch.where(self.count > 0, self.error_sum / self.count, torch.tensor(0.0))
    
    def update(self, sensed_temperature: Tensor, displayed_rgb: Optional[Tensor] = None):
        """
        Update metric with temperature and color data.
        
        Args:
            sensed_temperature : Actual temperatures sensed by agents [N]
            displayed_rgb      : RGB colors displayed by agents [N, 3] or None
        """
        if sensed_temperature.dim() > 1:
            sensed_temperature = sensed_temperature.squeeze(-1)
            
        if displayed_rgb is None:
            displayed_rgb = self._temperature_to_rgb(sensed_temperature)
            
        reconstructed_temp = self._rgb_to_temperature(displayed_rgb)
        error              = (sensed_temperature - reconstructed_temp).abs()
        
        self.error_sum += error.sum()
        self.count     += sensed_temperature.numel()


class EnergyConsumptionMetric(Metric):
    """
    Estimates power consumption based on control inputs.
    
    Uses a simplified quadrotor power model:
    P ∝ ||u_safe - g||^k
    
    where:
        - u_safe : safety-filtered control vector
        - g : gravity vector pointing downward
        - k : power exponent (typically 1.5 for quadrotors)
    """
    
    def __init__(self, gravity: float = 9.81, power_exponent: float = 1.5):
        """
        Initialize the energy metric.
        
        Args:
            gravity        : Gravitational acceleration (m/s²)
            power_exponent : Exponent k in the power model
        """
        super().__init__()
        self.gravity        = gravity
        self.power_exponent = power_exponent
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("power_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
    
    def compute(self) -> Tensor:
        """
        Compute the average power consumption per agent.
        
        Returns:
            Average power consumption in arbitrary units
        """
        return torch.where(self.count > 0, self.power_sum / self.count, torch.tensor(0.0))
    
    def update(self, u_safe: Tensor):
        """
        Computes instantaneous power from thrust vector magnitude.
        
        Args:
            u_safe : Safety-filtered control actions [N, 3] (m/s²)
        """
        gravity_vec = torch.zeros_like(u_safe)
        gravity_vec[..., 2] = -self.gravity
        
        thrust = u_safe - gravity_vec
        thrust_magnitude = thrust.norm(dim=-1)
        power = thrust_magnitude.pow(self.power_exponent)
        
        self.power_sum += power.sum()
        self.count     += u_safe.shape[0]


class LegibilitySSIMMetric(Metric):
    """
    Computes Structural Similarity Index (SSIM) between swarm and wind fields.
    
    This metric measures how well the swarm's collective motion pattern matches
    the underlying wind field, quantifying the visual "legibility" of the display.
    Uses kernel density estimation to render velocity fields onto 2D grids.
    """
    
    def __init__(self, grid_size: int = 64, sigma: float = 2.0):
        """
        Initialize the legibility metric.
        
        Args:
            grid_size : Size of the grid for rendering velocity fields
            sigma     : Standard deviation for Gaussian kernel in KDE
        """
        super().__init__()
        self.grid_size = grid_size
        self.sigma     = sigma
        
        self.ssim_metric = StructuralSimilarityIndexMeasure(
            data_range=1.0,
            kernel_size=11,
            reduction='elementwise_mean'
        )
        
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("ssim_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
    
    def _render_velocity_field(
        self, 
        bounds_min : Tensor,
        bounds_max : Tensor,
        positions  : Tensor, 
        velocities : Tensor
    ) -> Tensor:
        """
        Projects 3D positions and velocities onto the x-y plane and applies
        Gaussian kernel density estimation to create a smooth velocity field.
        
        Args:
            bounds_min : Minimum workspace bounds [3]
            bounds_max : Maximum workspace bounds [3]
            positions  : Agent positions [N, 3]
            velocities : Agent velocities [N, 3]
            
        Returns:
            2D tensor representing the velocity magnitude field
        """
        device = positions.device
        
        pos_2d = positions[:, :2]
        vel_magnitude = velocities[:, :2].norm(dim=1)
        
        bounds_min_2d = bounds_min[:2]
        bounds_max_2d = bounds_max[:2]
        norm_pos = (pos_2d - bounds_min_2d) / (bounds_max_2d - bounds_min_2d)
        grid_pos = norm_pos * (self.grid_size - 1)
        
        field = torch.zeros((self.grid_size, self.grid_size), device=device)
        
        grid_x, grid_y = torch.meshgrid(
            torch.arange(self.grid_size, device=device),
            torch.arange(self.grid_size, device=device),
            indexing='xy'
        )
        
        for i in range(positions.shape[0]):
            gx, gy = grid_pos[i]
            dist_sq = (grid_x - gx)**2 + (grid_y - gy)**2
            weight = torch.exp(-dist_sq / (2 * self.sigma**2))
            field += vel_magnitude[i] * weight
        
        field_max = field.max()
        if field_max > 0:
            field = field / field_max
            
        return field
    
    def compute(self) -> Tensor:
        """
        Compute the average SSIM score.
        
        Returns:
            Average SSIM score in range [0, 1], where 1 indicates perfect match
        """
        return torch.where(self.count > 0, self.ssim_sum / self.count, torch.tensor(0.0))
    
    def update(
        self, 
        bounds_min : Tensor,
        bounds_max : Tensor,
        positions  : Tensor, 
        velocities : Tensor,
        wind_field : Tensor
    ):
        """
        Computes a simplified SSIM using luminance and contrast terms:
        
        SSIM = (2*μ_x*μ_y + C1)(2*σ_xy + C2) / ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))
        
        where:
            - μ_x, μ_y : mean values of swarm and wind fields
            - σ_x, σ_y : standard deviations of fields
            - σ_xy : covariance between fields
            - C1, C2 : small constants to avoid division by zero
        
        Args:
            bounds_min : Minimum workspace bounds [3]
            bounds_max : Maximum workspace bounds [3]
            positions  : Agent positions [N, 3]
            velocities : Agent velocities [N, 3]
            wind_field : Ground truth wind velocities [N, 3]
        """
        swarm_field = self._render_velocity_field(
            bounds_min, bounds_max, positions, velocities
        )
        wind_field_rendered = self._render_velocity_field(
            bounds_min, bounds_max, positions, wind_field
        )
        
        swarm_field_img = swarm_field.unsqueeze(0).unsqueeze(0)
        wind_field_img = wind_field_rendered.unsqueeze(0).unsqueeze(0)
        
        ssim_value = self.ssim_metric(swarm_field_img, wind_field_img)
        
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
        max_temperature : float,
        output_dim      : int,
        gravity         : float = 9.81,
        grid_size       : int = 64
    ):
        """
        Initialize the metrics collector with all metric instances.
        
        Args:
            max_temperature : Maximum safe temperature threshold from config
            output_dim      : Dimension of the policy output
            gravity         : Gravitational acceleration for energy metric
            grid_size       : Grid size for SSIM computation
        """
        self.max_temperature = max_temperature
        self.output_dim      = output_dim
        
        self._init_imitation_metrics(output_dim)
        self._init_evaluation_metrics(max_temperature, gravity, grid_size)
        self._init_runtime_trackers()
    
    def _init_evaluation_metrics(
        self, 
        max_temperature : float,
        gravity         : float,
        grid_size       : int
    ):
        """
        Creates TorchMetrics instances for all five core performance metrics.
        
        Args:
            max_temperature : Temperature threshold for safety violations
            gravity         : Gravitational constant for energy calculations
            grid_size       : Resolution for SSIM field rendering
        """
        self.train_evaluation = MetricCollection({
            "avg_power_consumption"  : EnergyConsumptionMetric(gravity),
            "cohesion_connectivity"  : CohesionMetric(),
            "color_accuracy_mae"     : ColorAccuracyMetric(),
            "legibility_ssim"        : LegibilitySSIMMetric(grid_size),
            "thermal_violation_rate" : ThermalSafetyMetric(max_temperature),
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
            "rmse" : MeanSquaredError(squared=False),
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
        return self.cbf_activation_count / max(self.total_steps, 1)
    
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
        if loss is not None:
            module.log(
                name     = f"{phase}/loss",
                on_epoch = True,
                on_step  = phase == "train",
                prog_bar = True,
                value    = loss
            )
        
        imitation_metrics = (self.train_imitation if phase == "train" 
                           else self.val_imitation)
        module.log_dict(
            dictionary = imitation_metrics,
            on_epoch   = True,
            on_step    = phase == "train",
            prog_bar   = False
        )
        
        eval_metrics = (self.train_evaluation if phase == "train" 
                       else self.val_evaluation)
        module.log_dict(
            dictionary = eval_metrics,
            on_epoch   = True,
            on_step    = False,
            prog_bar   = False
        )
        
        if predictions is not None and targets is not None:
            dim_names = ["x", "y", "z"][:self.output_dim]
            for i, dim in enumerate(dim_names):
                dim_error = (predictions[..., i] - targets[..., i]).pow(2).mean()
                module.log(
                    name     = f"{phase}/velocity_{dim}_mse",
                    on_epoch = True,
                    value    = dim_error
                )
        
        if phase == "train":
            cbf_rate = self.get_cbf_activation_rate()
            module.log(
                name     = "train/cbf_activation_rate",
                on_epoch = True,
                on_step  = False,
                value    = cbf_rate
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
    
    def update_evaluation_metrics(self, batch: TensorDict, phase: str):
        """
        Extracts relevant fields from the batch and updates each metric
        appropriately. Handles missing fields gracefully.
        
        Args:
            batch : TensorDict containing simulation state and actions
            phase : Training phase ("train" or "val")
        """
        metrics = self.train_evaluation if phase == "train" else self.val_evaluation
        
        if "temperature" in batch:
            metrics["thermal_violation_rate"].update(batch["temperature"])
            metrics["color_accuracy_mae"].update(batch["temperature"])
        
        if all(k in batch for k in ["position", "velocity", "wind"]):
            device = batch["position"].device
            bounds_min = torch.zeros(3, device=device)
            bounds_max = torch.tensor([50.0, 50.0, 20.0], device=device)
            metrics["legibility_ssim"].update(
                bounds_min,
                bounds_max,
                batch["position"],
                batch["velocity"], 
                batch["wind"]
            )
        
        if "edge_index" in batch:
            num_agents = batch["position"].shape[0] if "position" in batch else 0
            metrics["cohesion_connectivity"].update(batch["edge_index"], num_agents)
        
        if "u_safe" in batch:
            metrics["avg_power_consumption"].update(batch["u_safe"])
        elif "action" in batch:
            metrics["avg_power_consumption"].update(batch["action"])
    
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
        metrics = self.train_imitation if phase == "train" else self.val_imitation
        metrics.update(predictions, targets)


class ThermalSafetyMetric(Metric):
    """
    Tracks the rate of thermal safety violations P(T_agent > T_max).
    
    This metric monitors how often agents exceed the maximum safe temperature
    threshold, which is critical for mission success and agent survival.
    Violations indicate failures of the safety system.
    """
    
    def __init__(self, max_temperature: float):
        """
        Initialize the thermal safety metric.
        
        Args:
            max_temperature : Maximum safe temperature threshold (Kelvin)
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
        return torch.where(self.total > 0, self.violations / self.total, torch.tensor(0.0))
    
    def update(self, temperature: Tensor):
        """
        Update metric with new temperature readings.
        
        Args:
            temperature : Agent temperatures [N] or [N, 1] in Kelvin
        """
        if temperature.dim() > 1:
            temperature = temperature.squeeze(-1)
            
        violations       = (temperature > self.max_temperature).sum(dtype=torch.float32)
        self.violations += violations
        self.total      += temperature.numel()