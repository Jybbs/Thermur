"""
Unified metrics collection for imitation learning training and evaluation.

This module provides a centralized MetricsCollector that manages all metrics
for the training pipeline, including imitation learning losses, core evaluation
metrics, and runtime performance tracking. The collector integrates seamlessly
with PyTorch Lightning's logging system and Weights & Biases.
"""
from __future__         import annotations
from torchmetrics       import MeanAbsoluteError, MeanSquaredError
from torchmetrics       import Metric, MetricCollection, R2Score
from torchmetrics.image import StructuralSimilarityIndexMeasure
from typing             import TYPE_CHECKING

if TYPE_CHECKING:
    from config.imitation.monitoring import MetricsModel
    from pytorch_lightning           import LightningModule
    from tensordict                  import TensorDictBase
    from torch                       import Tensor

import torch as th


class AveragingMetric(Metric):
    """
    Base class for metrics that compute running averages.
    
    Provides common sum/count state management and averaging logic
    for metrics that accumulate values over batches. Subclasses should
    implement the update() method to add values to the sum.
    """
    count : Tensor  # Number of measurements
    sum   : Tensor  # Sum of measured values
    
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
        if self.count > 0:
            return self.sum / self.count
        return th.zeros_like(self.sum)


class CohesionMetric(AveragingMetric):
    """
    Measures graph connectivity via the second smallest eigenvalue (λ₂).
    
    The algebraic connectivity (Fiedler value) quantifies how well-connected
    the communication graph is, with higher values indicating stronger cohesion.
    A disconnected graph has λ₂ = 0, while a complete graph has maximum λ₂.
    
    State variables:
    - count : Number of graph measurements
    - sum   : Sum of λ₂ values across all graphs
    """
    
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
            self.sum += 0.0
            self.count += 1
            return
        
        adj_matrix = th.zeros((num_agents, num_agents), device=edge_index.device)
        adj_matrix.index_put_(
            (edge_index[0], edge_index[1]), 
            th.ones(edge_index.shape[1], device=edge_index.device)
        )
        adj_matrix = ((adj_matrix + adj_matrix.T) > 0).float()
        laplacian  = th.diag(adj_matrix.sum(dim=1)) - adj_matrix
        
        try:
            eigenvalues    = th.linalg.eigvalsh(laplacian)
            lambda_squared = (eigenvalues[1] 
                              if eigenvalues.numel() > 1 
                              else eigenvalues.new_zeros(()))
        except RuntimeError:
            lambda_squared = laplacian.new_zeros(())
            
        self.sum += lambda_squared
        self.count += 1


class ColorAccuracyMetric(AveragingMetric):
    """
    Measures the Mean Absolute Error (mae_color) between sensed and displayed temperatures.
    
    This metric quantifies how accurately the swarm can display temperature
    information through their RGB LEDs, using a heat colormap mapping where
    blue represents cold and red represents hot temperatures.
    
    State variables:
    - count : Number of temperature measurements  
    - sum   : Sum of absolute temperature display errors (Kelvin)
    """
    temp_max : Tensor
    temp_min : Tensor
    
    def __init__(self, metrics: MetricsModel):
        """
        Initialize the color accuracy metric with temperature bounds.
        
        Sets up the temperature range for the heat colormap mapping,
        where colors interpolate from blue (cold) to red (hot).
        
        Args:
            metrics: Metrics configuration containing temperature bounds
        """
        super().__init__()
        self.register_buffer("temp_max", th.tensor(metrics.color_temp_max))
        self.register_buffer("temp_min", th.tensor(metrics.color_temp_min))
    
    def _temperature_to_rgb(self, temperature: Tensor) -> Tensor:
        """
        Map temperature to RGB color using heat colormap.
        
        Blue (cold) -> Green (medium) -> Red (hot)
        
        Args:
            temperature : Temperature values [N] in Kelvin
            
        Returns:
            RGB colors [N, 3] in range [0, 1]
        """
        temp_norm = th.clamp(
            (temperature - self.temp_min) / (self.temp_max - self.temp_min),
            0, 1
        )
        
        return th.stack([
            th.clamp(2 * temp_norm - 0.5, 0, 1),
            th.where(
                temp_norm < 0.5,
                2 * temp_norm,
                2 * (1 - temp_norm)
            ),
            th.clamp(1 - 2 * temp_norm, 0, 1)
        ], dim=-1)
    
    def update(
        self,
        sensed_temperature : Tensor,
        displayed_rgb      : Tensor | None = None
    ):
        """
        Update metric with temperature and color data.
        
        Args:
            sensed_temperature : Actual temperatures sensed by agents [N]
            displayed_rgb      : RGB colors displayed by agents [N, 3] or None
        """
        sensed_temperature = sensed_temperature.flatten()
        
        # TODO: Currently using computed RGB as placeholder until actual RGB display data is available
        if displayed_rgb is None:
            displayed_rgb = self._temperature_to_rgb(sensed_temperature)
            
        # Reconstruct temperature from RGB using red channel as indicator
        reconstructed_temp = th.lerp(self.temp_min, self.temp_max, displayed_rgb[..., 0])
        error = (sensed_temperature - reconstructed_temp).abs()
        
        self.sum   += error.sum()
        self.count += sensed_temperature.numel()


class EnergyConsumptionMetric(AveragingMetric):
    """
    Estimates average power consumption based on control inputs.
    
    Uses a simplified quadrotor power model:
    P ∝ ||u_safe - g||^k
    
    where:
        - u_safe : safety-filtered control vector
        - g : gravity vector pointing downward
        - k : power exponent (typically 1.5 for quadrotors)
        
    State variables:
    - count : Number of control action measurements
    - sum   : Sum of power estimates P = ||u - g||^k
    """
    gravity: Tensor
    
    def __init__(
        self,
        gravity : float,
        metrics : MetricsModel
    ):
        """
        Initialize the energy metric with physics parameters.
        
        Sets up the gravity constant and power exponent for the
        quadrotor power consumption model. The power is proportional
        to the thrust magnitude relative to gravity.
        
        Args:
            gravity : Gravitational acceleration from physics config
            metrics : Metrics configuration containing power exponent
        """
        super().__init__()
        self.register_buffer("gravity", th.tensor(gravity))
        self.power_exponent = metrics.power_exponent
    
    def update(self, u_safe: Tensor):
        """
        Computes instantaneous power from thrust vector magnitude.
        
        Args:
            u_safe : Safety-filtered control actions [N, 3] (m/s²)
        """
        gravity_vec = th.zeros_like(u_safe)
        gravity_vec[..., 2] = -self.gravity
        
        power = (u_safe - gravity_vec).norm(dim=-1).pow(self.power_exponent)
        
        self.sum   += power.sum()
        self.count += u_safe.shape[0]


class LegibilitySSIMMetric(AveragingMetric):
    """
    Computes Structural Similarity Index Measure (ssim) between swarm and wind fields.
    
    This metric measures how well the swarm's collective motion pattern matches
    the underlying wind field, quantifying the visual "legibility" of the display.
    Uses kernel density estimation to render velocity fields onto 2D grids.
    
    State variables:
    - count : Number of SSIM measurements
    - sum   : Sum of SSIM scores (0-1 range per measurement)
    """
    bounds_max : Tensor
    bounds_min : Tensor
    coords     : Tensor
    
    def __init__(
        self,
        bounds_max : list[float],
        metrics    : MetricsModel
    ):
        """
        Initialize the legibility metric with rendering parameters.
        
        Sets up the grid resolution and kernel parameters for converting
        discrete agent positions and velocities into continuous fields
        that can be compared using SSIM.
        
        Args:
            bounds_max : Maximum workspace bounds from physics config
            metrics    : Metrics configuration with grid and kernel settings
        """
        super().__init__()
        self.grid_size   = metrics.legibility_grid_size
        self.kernel_size = metrics.legibility_kernel_size
        self.sigma       = metrics.legibility_sigma
        
        self.ssim_metric = StructuralSimilarityIndexMeasure(
            data_range  = 1.0,
            kernel_size = self.kernel_size,
            reduction   = 'elementwise_mean'
        )
        
        self.register_buffer("bounds_max", th.tensor(bounds_max))
        self.register_buffer("bounds_min", th.zeros(3))
        self.register_buffer("coords", self._pre_compute_coordinates(self.grid_size))
    
    def _pre_compute_coordinates(self, grid_size: int) -> Tensor:
        """
        Pre-compute the coordinate grid for rendering.
        
        Creates a 2D grid of coordinates that will be used for
        kernel density estimation when rendering velocity fields.
        
        Args:
            grid_size : Resolution of the grid
            
        Returns:
            Coordinate tensor of shape [grid_size, grid_size, 2]
        """
        return th.stack(
            th.meshgrid(
                th.arange(grid_size),
                th.arange(grid_size),
                indexing = 'xy'
            ), 
            dim = -1
        ).float()
    
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
        pos_2d = positions[:, :2]
        vel_magnitude = velocities[:, :2].norm(dim=1)
        
        grid_pos = ((pos_2d - bounds_min[:2]) / 
                    (bounds_max[:2] - bounds_min[:2]) * 
                    (self.grid_size - 1))
        
        field = th.zeros((self.grid_size, self.grid_size), device=positions.device)
        
        grid_pos_expanded = grid_pos.unsqueeze(0).unsqueeze(0)
        coords_expanded   = self.coords.unsqueeze(2)
        
        dist_sq = ((coords_expanded - grid_pos_expanded) ** 2).sum(dim=-1)
        weights = th.exp(-dist_sq / (2 * self.sigma**2))
        field   = (weights * vel_magnitude.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        
        field_max = field.max()
        return field / field_max if field_max > 0 else field
    
    def update(
        self,
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
            - σ_xy     : covariance between fields
            - C1, C2   : small constants to avoid division by zero
        
        Args:
            positions  : Agent positions [N, 3]
            velocities : Agent velocities [N, 3]
            wind_field : Ground truth wind velocities [N, 3]
        """
        swarm_field = self._render_velocity_field(
            self.bounds_min, self.bounds_max, positions, velocities
        ).unsqueeze(0).unsqueeze(0)
        
        wind_field_rendered = self._render_velocity_field(
            self.bounds_min, self.bounds_max, positions, wind_field
        ).unsqueeze(0).unsqueeze(0)
        
        ssim_value = self.ssim_metric(swarm_field, wind_field_rendered)
        
        self.sum   += ssim_value
        self.count += 1


class MetricsCollector:
    """
    Centralized metric collection and management for training and evaluation.
    
    This class owns all TorchMetrics instances and provides methods to update
    and log metrics throughout the training process. It handles:
    
    1. Imitation learning metrics (MSE, RMSE, MAE, R²)
    2. Core evaluation metrics (legibility, cohesion, energy, color)
    
    The collector is designed to work with PyTorch Lightning's logging system
    and automatically syncs metrics to Weights & Biases through the configured
    logger.
    """
    def __init__(
        self,
        bounds_max      : list[float],
        gravity         : float,
        max_temperature : float,
        metrics         : MetricsModel
    ):
        """
        Initialize the metrics collector with all metric instances.
        
        Args:
            bounds_max      : Maximum workspace bounds from physics config
            gravity         : Gravitational acceleration from physics config
            max_temperature : Maximum safe temperature from flock config
            metrics         : Metrics configuration model
        """
        self.bounds_max      = bounds_max
        self.gravity         = gravity
        self.max_temperature = max_temperature
        self.metrics         = metrics
        
        self._init_imitation_metrics()
        self._init_evaluation_metrics()
    
    def _get_metrics(self, metric_type: str, is_training: bool) -> MetricCollection:
        """
        Get the appropriate metrics collection based on type and phase.
        
        Args:
            metric_type : Either "imitation" or "evaluation"
            is_training : Whether this is training (True) or validation (False)
            
        Returns:
            The corresponding MetricCollection instance
        """
        return getattr(
            self, 
            f"{("train" if is_training else "val")}_{metric_type}"
        )
    
    def _init_evaluation_metrics(self):
        """
        Initialize the four core evaluation metrics that assess swarm performance.
        
        These metrics evaluate how well the swarm achieves its mission objectives:
        - λ₂ (Cohesion)     : Algebraic connectivity measuring flock cohesion
        - SSIM (Legibility) : How well flock motion visualizes the wind field  
        - MAE Color         : Accuracy of temperature display through RGB LEDs
        - Average Power     : Energy consumption for mission endurance
        
        Each metric is wrapped in MetricCollection for automatic train/val splitting
        and distributed training synchronization.
        """
        self.train_evaluation = MetricCollection({
            "avg_power"  : EnergyConsumptionMetric(self.gravity, self.metrics),
            "λ₂"         : CohesionMetric(),
            "mae_color"  : ColorAccuracyMetric(self.metrics),
            "ssim"       : LegibilitySSIMMetric(self.bounds_max, self.metrics),
        })
        self.val_evaluation = self.train_evaluation.clone(prefix="val_")
    
    def _init_imitation_metrics(self):
        """
        Initialize metrics for evaluating imitation learning performance.
        
        These regression metrics measure how well the neural network policy
        mimics the expert controller's behavior:
        - MSE  : Mean squared error of velocity predictions
        - RMSE : Root mean squared error for interpretable units
        - MAE  : Mean absolute error for robustness to outliers
        - R²   : Coefficient of determination for overall fit quality
        
        All metrics track 3D velocity predictions (x, y, z).
        """
        self.train_imitation = MetricCollection({
            "mae"  : MeanAbsoluteError(),
            "mse"  : MeanSquaredError(),
            "r2"   : R2Score(num_outputs=3),
            "rmse" : MeanSquaredError(False),
        })
        self.val_imitation = self.train_imitation.clone(prefix="val_")
    
    def log_all_metrics(
        self,
        is_training : bool,
        module      : LightningModule,
        step_output : bool,
        loss        : Tensor | None = None,
        predictions : Tensor | None = None,
        targets     : Tensor | None = None
    ):
        """
        Log all metrics to PyTorch Lightning and external loggers.
        
        Orchestrates the logging of different metric categories with appropriate
        frequencies and visualization settings. Imitation metrics are logged at
        every training step for close monitoring of learning progress, while
        evaluation metrics are logged only at epoch boundaries to reduce noise.
        
        Special handling includes:
        - Loss displayed in progress bar for immediate feedback
        - Per-dimension velocity MSE for debugging specific axes
        - Automatic train/val prefixing for metric organization
        
        Args:
            is_training : Whether this is training (True) or validation (False)
            module      : Lightning module providing the logger interface
            step_output : Whether this is step-level output or epoch-end aggregation
            loss        : Behavioral cloning loss    (required if step_output=True)
            predictions : Model velocity predictions (required if step_output=True)
            targets     : Expert velocity commands   (required if step_output=True)
        """
        phase = "train" if is_training else "val"
        
        if step_output:
            assert loss        is not None, "Loss required for step logging"
            assert predictions is not None, "Predictions required for step logging"
            assert targets     is not None, "Targets required for step logging"
            
            module.log(
                name     = f"{phase}/loss",
                value    = loss,
                on_epoch = True,
                on_step  = is_training,
                prog_bar = True
            )
            
            for i, dim in enumerate(["x", "y", "z"]):
                module.log(
                    name     = f"{phase}/velocity_{dim}_mse",
                    value    = (predictions[..., i] - targets[..., i]).pow(2).mean(),
                    on_epoch = True,
                    on_step  = False
                )
        
        for metric_type, on_step in [
            ("imitation",  is_training), 
            ("evaluation", False)
        ]:
            module.log_dict(
                dictionary = self._get_metrics(metric_type, is_training),
                on_epoch   = True,
                on_step    = on_step
            )
        
    def update_evaluation_metrics(
        self,
        batch       : TensorDictBase,
        is_training : bool
    ):
        """
        Update all evaluation metrics from a single simulation batch.
        
        Intelligently extracts relevant data for each metric based on what
        fields are available in the batch. This allows graceful handling of
        partial data during different training stages or evaluation modes.
        
        Field requirements by metric:
        - Color accuracy  : temperature
        - Legibility SSIM : position, velocity, wind  
        - Cohesion λ₂     : edge_index, position (for agent count)
        - Energy          : u_safe or action (control inputs)
        
        Args:
            batch       : TensorDict containing simulation state and actions
            is_training : Whether this is training (True) or validation (False)
        """
        metrics = self._get_metrics("evaluation", is_training)
        
        if "temperature" in batch:
            metrics["mae_color"].update(batch["temperature"])
        
        if all(k in batch for k in ["position", "velocity", "wind"]):
            metrics["ssim"].update(
                positions  = batch["position"],
                velocities = batch["velocity"],
                wind_field = batch["wind"]
            )
        
        if "edge_index" in batch:
            num_agents = batch["position"].shape[0] if "position" in batch else 0
            metrics["λ₂"].update(
                edge_index = batch["edge_index"], 
                num_agents = num_agents
            )
        
        if u_control := batch.get("u_safe") or batch.get("action"):
            metrics["avg_power"].update(u_safe=u_control)
    
    def update_imitation_metrics(
        self,
        is_training : bool,
        predictions : Tensor,
        targets     : Tensor
    ):
        """
        Update regression metrics measuring behavioral cloning accuracy.
        
        Computes multiple error metrics between the neural network's velocity
        predictions and the expert controller's demonstrated actions. These
        metrics guide the imitation learning process and help diagnose
        prediction quality across different error characteristics.
        
        Args:
            is_training : Whether this is training (True) or validation (False)
            predictions : Model velocity outputs [batch_size, 3] in m/s
            targets     : Expert velocity commands [batch_size, 3] in m/s
        """
        self._get_metrics("imitation", is_training).update(predictions, targets)
