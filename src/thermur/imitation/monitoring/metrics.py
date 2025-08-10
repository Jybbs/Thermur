"""
Unified metrics collection for imitation learning training and evaluation.

This module provides a centralized MetricsCollector that manages all metrics
for the training pipeline, including imitation learning losses, core evaluation
metrics, and runtime performance tracking. The collector integrates seamlessly
with PyTorch Lightning's logging system and Weights & Biases.
"""
from __future__         import annotations
from itertools          import pairwise
from tensordict         import TensorDictBase
from torchmetrics       import MeanAbsoluteError, MeanSquaredError
from torchmetrics       import Metric, MetricCollection, R2Score
from torchmetrics.image import StructuralSimilarityIndexMeasure
from typing             import TYPE_CHECKING

if TYPE_CHECKING:
    from config.imitation.controller import MurmurationModel, SafetyModel
    from config.imitation.monitoring import MetricsModel
    from config.types                import StepMetrics
    from pytorch_lightning           import LightningModule
    from torch                       import Tensor

import torch as th


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


class CohesionMetric(AveragingMetric):
    """
    Measure graph connectivity via the Fiedler value λ₂.

    The algebraic connectivity quantifies how well-connected the flock's
    communication graph is. Higher values indicate stronger cohesion, with
    λ₂ = 0 for disconnected graphs and λ₂ > 0 for connected components.

    The metric computes the second-smallest eigenvalue of the graph Laplacian:

        L = D - A

    where D is the degree matrix and A is the adjacency matrix.
    """

    def _compute_graph_laplacian(
        self,
        edge_index : Tensor,
        num_agents : int
    ) -> Tensor:
        """
        Construct the graph Laplacian matrix L = D - A.

        Builds a symmetric adjacency matrix from the edge list and computes
        the Laplacian for spectral analysis.

        Args:
            edge_index : Graph edges [2, E] in COO format  
            num_agents : Number of nodes in the graph

        Returns:
            Tensor [n, n] symmetric Laplacian matrix
        """
        adj_sparse = th.sparse_coo_tensor(
            indices = edge_index,
            values  = th.ones(edge_index.shape[1], device=edge_index.device),
            size    = (num_agents, num_agents)
        )
        adj_matrix = (adj_sparse + adj_sparse.t()).to_dense().bool().float()
        
        return th.diag_embed(adj_matrix.sum(1)) - adj_matrix

    def update(
        self,
        edge_index : Tensor,
        num_agents : int
    ):
        """
        Update metric with graph connectivity measurement.

        Computes the Fiedler value (second-smallest eigenvalue) of the
        graph Laplacian to quantify algebraic connectivity.

        Args:
            edge_index : Graph connectivity tensor [2, E] or [B, 2, E] in COO format
            num_agents : Total number of agents in the graph
        """
        if edge_index.numel() == 0:
            self.sum   += 0.0
            self.count += 1
            return

        if edge_index.dim() == 3:
            for i in range(edge_index.shape[0]):
                self.update(edge_index[i], num_agents)
            return

        laplacian = self._compute_graph_laplacian(edge_index, num_agents)

        try:
            eigenvalues = th.linalg.eigvalsh(laplacian)
            self.sum   += eigenvalues[1] if len(eigenvalues) > 1 else 0.0
        except RuntimeError:
            self.sum   += 0.0
        
        self.count += 1


class ColorAccuracyMetric(AveragingMetric):
    """
    Measures the Mean Absolute Error between sensed and displayed temperatures.

    This metric quantifies how accurately the flock can display temperature
    information through their RGB LEDs, using a heat colormap mapping where
    blue represents cold and red represents hot temperatures.

    The colormap follows:
        T_min → Blue (0,0,1)
        T_mid → Green (0,1,0)
        T_max → Red (1,0,0)
    """
    temp_max : Tensor
    temp_min : Tensor

    def __init__(self, metrics: MetricsModel):
        """
        Initialize the color accuracy metric with temperature bounds.

        Args:
            metrics : Metrics configuration containing temperature bounds
        """
        super().__init__()
        self.register_buffer("temp_max", th.tensor(metrics.color_temp_max))
        self.register_buffer("temp_min", th.tensor(metrics.color_temp_min))

    def _temperature_to_rgb(self, temperature: Tensor) -> Tensor:
        """
        Map temperature to RGB color using heat colormap.

        Implements piecewise linear interpolation:
            - R channel: 0 at T_min, 1 at T_max
            - G channel: Peaks at T_mid
            - B channel: 1 at T_min, 0 at T_max

        Args:
            temperature : Temperature values [N] in Kelvin

        Returns:
            RGB colors [N, 3] in range [0, 1]
        """
        normalized_temp = (
            (temperature - self.temp_min) / 
            (self.temp_max - self.temp_min)
        ).clamp(0, 1)

        return th.stack([
            th.clamp(2 * normalized_temp - 0.5, 0, 1),
            th.where(
                normalized_temp < 0.5,
                2 * normalized_temp,
                2 * (1 - normalized_temp)
            ),
            th.clamp(1 - 2 * normalized_temp, 0, 1)
        ], dim=-1)

    def update(
        self,
        displayed_rgb      : Tensor | None,
        sensed_temperature : Tensor
    ):
        """
        Update metric with temperature and color data.

        Args:
            displayed_rgb      : RGB colors displayed by agents [N, 3] or None
            sensed_temperature : Actual temperatures sensed by agents [N]
        """
        sensed_temperature = sensed_temperature.flatten()

        if displayed_rgb is None:
            displayed_rgb = self._temperature_to_rgb(sensed_temperature)

        if (n_temps := sensed_temperature.numel()) > 0:
            reconstructed_temp = th.lerp(
                self.temp_min, self.temp_max, displayed_rgb[..., 0]
            )
            self.sum   += (sensed_temperature - reconstructed_temp).abs().sum()
            self.count += n_temps


class DynamicBalanceMetric(AveragingMetric):
    """
    Measures learned flock's density balance under thermal threat.

    Tracks the ratio of expansion to contraction when the learned policy
    encounters high temperatures, revealing whether it maintains the "ink-like"
    evasion pattern characteristic of murmurations. The balance ratio:
    
        β = r_avg / d_avg

    where r_avg is average distance to center and d_avg is average pairwise
    distance. Ideal range is [0.5, 2.0] for balanced expansion/contraction.
    """
    
    def __init__(self, safety: SafetyModel):
        """
        Initialize with thermal threat threshold.
        
        Args:
            safety: Safety model with temperature thresholds
        """
        super().__init__()
        self.threat_temperature = safety.max_temperature * safety.threat_ratio
    
    def update(self, batch: TensorDictBase):
        """
        Update metric with density ratio when under thermal threat.

        Args:
            batch: TensorDict containing position and temperature
        """
            
        if batch["temperature"].max() <= self.threat_temperature:
            return

        center   = batch["position"].mean(dim=0)
        pairwise = th.cdist(batch["position"], batch["position"]).triu(1)
        
        if (n_pairs := pairwise.gt(0).sum()) > 0:
            self.sum   += (
                (batch["position"] - center).norm(dim=1).mean() / 
                (pairwise.sum() / n_pairs).clamp_min(1e-8)
            )
            self.count += 1


class EnergyConsumptionMetric(AveragingMetric):
    """
    Estimates average power consumption based on control inputs.

    Uses a simplified quadrotor power model from Hoffmann et al. (2011):
        P ∝ ||u - g||^k

    where:
        - u : control acceleration vector (m/s²)
        - g : gravity vector pointing downward
        - k : power exponent (typically 1.5 for quadrotors)
    """
    gravity: Tensor

    def __init__(
        self,
        gravity : float,
        metrics : MetricsModel
    ):
        """
        Initialize the energy metric with physics parameters.

        Args:
            gravity : Gravitational acceleration (m/s²)
            metrics : Metrics configuration containing power exponent k
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
        gravity_vector         = th.zeros_like(u_safe)
        gravity_vector[..., 2] = -self.gravity
        thrust_magnitude       = (u_safe - gravity_vector).norm(dim=-1)

        self.sum   += thrust_magnitude.pow(self.power_exponent).sum()
        self.count += u_safe.shape[0]


class LegibilitySSIMMetric(AveragingMetric):
    """
    Measure visual similarity between flock motion and wind field.

    Quantifies how well the flock's collective motion pattern matches
    the underlying wind field, assessing the "legibility" of the aerial
    display. Uses kernel density estimation to render velocity fields
    onto 2D grids for comparison.

    The Structural Similarity Index is computed as:

        SSIM = (2μ_xμ_y + C₁)(2σ_xy + C₂) / ((μ_x² + μ_y² + C₁)(σ_x² + σ_y² + C₂))

    where μ, σ are local means and variances, and C₁, C₂ are stability constants.
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
        Initialize legibility metric with rendering parameters.

        Sets up the SSIM computation with kernel density estimation
        parameters for rendering velocity fields onto 2D grids.

        Args:
            bounds_max : Maximum workspace bounds [x_max, y_max, z_max]
            metrics    : Configuration with grid size and kernel parameters
        """
        super().__init__()
        self.grid_size    = metrics.legibility_grid_size
        self.kernel_size  = metrics.legibility_kernel_size
        self.sigma        = metrics.legibility_sigma
        self.ssim_metric  = StructuralSimilarityIndexMeasure(
            data_range  = 1.0,
            kernel_size = self.kernel_size,
            reduction   = 'elementwise_mean'
        )

        self.register_buffer("bounds_max", th.tensor(bounds_max))
        self.register_buffer("bounds_min", th.zeros(3))
        self.register_buffer("coords", self._pre_compute_coordinates(self.grid_size))

    def _pre_compute_coordinates(self, grid_size: int) -> Tensor:
        """
        Pre-compute 2D coordinate grid for KDE rendering.

        Creates a meshgrid of (x, y) coordinates used for evaluating
        Gaussian kernels during velocity field rendering.

        Args:
            grid_size: Resolution of the rendering grid

        Returns:
            Tensor [grid_size, grid_size, 2] of grid coordinates
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
        Render velocity field using Gaussian kernel density estimation.

        Projects 3D velocity data onto the x-y plane and applies Gaussian
        kernels to create a continuous field representation:

            f(𝐱) = Σᵢ |𝐯ᵢ| exp(-‖𝐱 - 𝐱ᵢ‖² / 2σ²)

        Args:
            bounds_max : Maximum workspace bounds [3]
            bounds_min : Minimum workspace bounds [3]  
            positions  : Agent positions 𝐱 ∈ ℝ^(n×3)
            velocities : Agent velocities 𝐯 ∈ ℝ^(n×3)

        Returns:
            Tensor [grid_size, grid_size] of normalized velocity magnitudes
        """
        velocity_magnitude = velocities[:, :2].norm(dim=1)
        grid_positions     = (
            (positions[:, :2] - bounds_min[:2]) /
            (bounds_max[:2]   - bounds_min[:2]) *
            (self.grid_size - 1)
        )

        distance_squared = th.cdist(self.coords.view(-1, 2), grid_positions).pow(2)
        kernel_weights   = th.exp(-distance_squared * (1.0 / (2 * self.sigma ** 2)))
        flattened        = (kernel_weights * velocity_magnitude).sum(dim=-1)
        field            = flattened.view(self.grid_size, self.grid_size)

        return field / field.max().clamp_min(1e-8)

    def update(
        self,
        positions  : Tensor,
        velocities : Tensor, 
        wind       : Tensor
    ):
        """
        Update metric with SSIM between flock and wind velocity fields.

        Renders both the flock's velocity field and the environmental wind
        field onto 2D grids, then computes their structural similarity.

        Args:
            positions  : Agent positions 𝐱 ∈ ℝ^(B×n×3)
            velocities : Agent velocities 𝐯 ∈ ℝ^(B×n×3)
            wind       : Environmental wind field 𝐰 ∈ ℝ^(B×n×3)
        """
        batch_size = positions.shape[0]
        total_ssim = 0.0
        
        for i in range(batch_size):
            flock_field = self._render_velocity_field(
                bounds_max = self.bounds_max,
                bounds_min = self.bounds_min,
                positions  = positions[i],
                velocities = velocities[i]
            )

            wind_field = self._render_velocity_field(
                bounds_max = self.bounds_max,
                bounds_min = self.bounds_min,
                positions  = positions[i],
                velocities = wind[i]
            )

            ssim_value = self.ssim_metric(
                preds  = flock_field.unsqueeze(0).unsqueeze(0),
                target = wind_field.unsqueeze(0).unsqueeze(0)
            )
            
            total_ssim += ssim_value
        
        self.sum   += total_ssim
        self.count += batch_size


class ScaleFreeCorrelationMetric(AveragingMetric):
    """
    Measure deviation from scale-free velocity correlations.
    
    Verifies that the flock exhibits power-law velocity correlations
    characteristic of critical systems. The correlation function C(r)
    should follow:

        C(r) ~ r^(-γ)

    where γ ≈ 1/3 for natural murmurations (Cavagna et al. 2010).
    """
    
    def __init__(self, mmm: MurmurationModel):
        """
        Initialize with target correlation exponent.
        
        Args:
            mmm: Murmuration model with expected exponent γ
        """
        super().__init__()
        self.target_exponent = mmm.correlation_exponent
    
    def _compute_velocity_correlations(
        self,
        positions  : Tensor,
        velocities : Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Compute pairwise velocity correlations and distances.

        Calculates the correlation function C(r) = ⟨δ𝐯ᵢ · δ𝐯ⱼ⟩ for all
        pairs of agents, where δ𝐯 = 𝐯 - ⟨𝐯⟩ are velocity fluctuations.

        Args:
            positions  : Agent positions 𝐱 ∈ ℝ^(n×3)
            velocities : Agent velocities 𝐯 ∈ ℝ^(n×3)

        Returns:
            Tuple of (correlation_matrix, distance_matrix)
        """
        distances  = th.cdist(positions, positions)
        spins      = th.nn.functional.normalize(velocities, dim=1)
        delta_spin = spins - spins.mean(dim=0, keepdim=True)
        corr_mat   = delta_spin @ delta_spin.mT
        
        return corr_mat, distances

    def _fit_power_law(
        self,
        bin_distances    : Tensor,
        bin_correlations : Tensor
    ) -> float:
        """
        Fit power law to binned correlation data.

        Uses log-log linear regression to estimate the exponent γ in:
            log C(r) = -γ log r + const

        Args:
            bin_distances    : Mean distance per bin [n_bins]
            bin_correlations : Mean correlation per bin [n_bins]

        Returns:
            Estimated power law exponent γ
        """
        log_r = bin_distances.log()
        log_c = bin_correlations.abs().clamp_min(1e-8).log()
        
        X = log_r - log_r.mean()
        Y = log_c - log_c.mean()
        
        return float(-(X @ Y) / (X @ X)) if (X @ X) > 0 else 0.0

    def update(self, batch: TensorDictBase):
        """
        Update metric with scale-free correlation measurement.
        
        Computes velocity correlations, bins by distance, and fits
        a power law to measure deviation from expected exponent.
        
        Args:
            batch: TensorDict containing position and velocity
        """
        corr_mat, distances = self._compute_velocity_correlations(
            batch["position"], batch["velocity"]
        )
        
        triu_mask = th.triu(th.ones_like(distances), diagonal=1).bool()
        if not triu_mask.any():
            return
            
        bin_edges = th.logspace(
            end   = distances[triu_mask].max().log10(),
            start = distances[triu_mask].min().log10(),
            steps = 11
        )
        
        bin_stats = [
            (corr_mat[mask].mean(), distances[mask].mean())
            for low, high in pairwise(bin_edges)
            if (mask := triu_mask & distances.gt(low) & distances.le(high)).any()
        ]
        
        if len(bin_stats) >= 3:
            bin_data = th.tensor(bin_stats)
            fitted_exponent = self._fit_power_law(
                bin_distances   = bin_data[:, 1],
                bin_correlations = bin_data[:, 0]
            )
            
            self.sum   += abs(fitted_exponent - self.target_exponent)
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
        bounds_max : list[float],
        gravity    : float,
        metrics    : MetricsModel,
        mmm        : MurmurationModel,
        safety     : SafetyModel
    ):
        """
        Initialize metrics collector with configuration parameters.

        Creates all metric instances for tracking imitation learning,
        evaluation, and murmuration-specific measurements.

        Args:
            bounds_max : Maximum workspace bounds [x_max, y_max, z_max]
            gravity    : Gravitational acceleration [m/s²]
            metrics    : Metrics configuration model
            mmm        : Murmuration dynamics configuration
            safety     : Safety configuration with temperature thresholds
        """
        self.bounds_max = bounds_max
        self.gravity    = gravity
        self.metrics    = metrics
        self.mmm        = mmm
        self.safety     = safety

        self._init_evaluation_metrics()
        self._init_imitation_metrics()

    def _compute_ready_metrics(
        self, 
        metrics: MetricCollection
    ) -> dict[str, Tensor] | None:
        """
        Compute metrics that have received data.
        
        Returns computed metrics if any have been updated, otherwise None.
        This prevents warnings during sanity check when compute() is called
        before any update() calls.
        
        Args:
            metrics: Collection of metrics to potentially compute
            
        Returns:
            Dictionary of computed values or None if no metrics have data
        """
        for metric in metrics.values():
            if isinstance(metric, AveragingMetric) and metric.count > 0:
                return metrics.compute()
            elif not isinstance(metric, AveragingMetric):
                return metrics.compute()
        return None

    def _get_metrics(
        self, 
        is_training : bool, 
        metric_type : str
    ) -> MetricCollection:
        """
        Get appropriate metrics collection by type and phase.

        Retrieves the correct MetricCollection instance based on whether
        we're in training/validation and which metric category is needed.

        Args:
            is_training : Whether in training (True) or validation (False) phase
            metric_type : Category name ("imitation", "evaluation", "murmuration")

        Returns:
            MetricCollection for the specified type and phase
        """
        return getattr(
            self,
            f"{("train" if is_training else "val")}_{metric_type}"
        )

    def _init_evaluation_metrics(self):
        """
        Initialize the four core evaluation metrics that assess flock performance.

        These metrics evaluate how well the flock achieves its mission objectives:
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

        These metrics include regression accuracy and spatial murmuration properties:
        - dynamic_balance  : Density ratio under thermal threat
        - MSE/RMSE/MAE/R²  : Regression metrics for velocity predictions
        - scale_free_error : Deviation from power-law correlations C(r) ~ r^(-1/3)

        All metrics work on individual frames without temporal dependencies.
        """
        create_imitation_metrics = lambda prefix: MetricCollection({
            f"{prefix}_dynamic_balance" : DynamicBalanceMetric(self.safety),
            f"{prefix}_mae"             : MeanAbsoluteError(),
            f"{prefix}_mse"             : MeanSquaredError(),
            f"{prefix}_r2"              : R2Score(multioutput='uniform_average'),
            f"{prefix}_rmse"            : MeanSquaredError(squared=False),
            f"{prefix}_scale_free"      : ScaleFreeCorrelationMetric(self.mmm)
        }, compute_groups=False)
        
        self.train_imitation = create_imitation_metrics("training")
        self.val_imitation   = create_imitation_metrics("validation")
    

    def log_all_metrics(
        self,
        is_training : bool,
        module      : LightningModule,
        step_data   : StepMetrics | None = None
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
            is_training : Whether in training (True) or validation (False) phase
            module      : Lightning module providing the logger interface
            step_data   : Optional step metrics (loss, predictions, targets) for
                          step-level logging. When None, only logs aggregated metrics.
        """
        phase = "training" if is_training else "validation"
        if step_data is not None:
            module.log(
                name     = f"{phase}/loss",
                on_epoch = True,
                on_step  = is_training,
                prog_bar = True,
                value    = step_data["loss"]
            )

            for i, dim in enumerate(["x", "y", "z"]):
                module.log(
                    name     = f"{phase}/velocity_{dim}_mse",
                    on_epoch = True,
                    on_step  = False,
                    value    = (step_data["predictions"][..., i] - 
                               step_data["targets"][..., i]).pow(2).mean()
                )
            
            if computed := self._compute_ready_metrics(
                self._get_metrics(is_training, "imitation")
            ):
                module.log_dict(
                    dictionary = computed,
                    on_epoch   = True,
                    on_step    = is_training
                )
        
        if not is_training or step_data is None:
            if computed := self._compute_ready_metrics(
                self._get_metrics(is_training, "evaluation")
            ):
                module.log_dict(
                    dictionary = computed,
                    on_epoch   = True,
                    on_step    = False
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
        metrics = self._get_metrics(is_training, "evaluation")

        if "temperature" in batch:
            metrics["mae_color"].update(
                displayed_rgb      = None,
                sensed_temperature = batch["temperature"]
            )

        if all(k in batch for k in ["position", "velocity", "wind"]):
            metrics["ssim"].update(
                positions  = batch["position"],
                velocities = batch["velocity"],
                wind       = batch["wind"]
            )

        if "edge_index" in batch and "position" in batch:
            num_agents = batch["position"].shape[1]
            
            metrics["λ₂"].update(
                edge_index = batch["edge_index"],
                num_agents = num_agents
            )

        u_control = batch.get("u_safe") if "u_safe" in batch else batch.get("action")
        if u_control is not None:
            metrics["avg_power"].update(u_safe=u_control)

    def update_imitation_metrics(
        self,
        batch       : TensorDictBase | None,
        is_training : bool,
        predictions : Tensor,
        targets     : Tensor
    ):
        """
        Update regression and spatial murmuration metrics.

        Computes both behavioral cloning accuracy and spatial murmuration properties
        that don't require temporal continuity.

        Args:
            batch       : Optional TensorDict with position, velocity, temperature
            is_training : Whether this is training (True) or validation (False)
            predictions : Model velocity outputs [batch_size, 3] in m/s
            targets     : Expert velocity commands [batch_size, 3] in m/s
        """
        metrics = self._get_metrics(is_training, "imitation")
        
        for name in ["mae", "mse", "r2", "rmse"]:
            key = f"training_{name}" if is_training else f"validation_{name}"
            if key in metrics:
                metrics[key].update(predictions, targets)
        
        if batch is not None:
            for name in ["scale_free", "dynamic_balance"]:
                key = f"training_{name}" if is_training else f"validation_{name}"
                if key in metrics:
                    metrics[key].update(batch)
    
