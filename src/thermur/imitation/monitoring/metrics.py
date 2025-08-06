"""
Unified metrics collection for imitation learning training and evaluation.

This module provides a centralized MetricsCollector that manages all metrics
for the training pipeline, including imitation learning losses, core evaluation
metrics, and runtime performance tracking. The collector integrates seamlessly
with PyTorch Lightning's logging system and Weights & Biases.
"""
from __future__         import annotations
from itertools          import combinations, pairwise
from torchmetrics       import MeanAbsoluteError, MeanSquaredError
from torchmetrics       import Metric, MetricCollection, R2Score
from torchmetrics.image import StructuralSimilarityIndexMeasure
from typing             import TYPE_CHECKING

if TYPE_CHECKING:
    from config.imitation.controller import MurmurationModel
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
            edge_index : Graph connectivity tensor [2, E] in COO format
            num_agents : Total number of agents in the graph
        """
        if edge_index.numel() == 0 or num_agents < 2:
            self.sum   += 0.0
            self.count += 1
            return

        laplacian = self._compute_graph_laplacian(edge_index, num_agents)

        try:
            eigenvalues    = th.linalg.eigvalsh(laplacian)
            lambda_squared = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
        except RuntimeError:
            lambda_squared = 0.0

        self.sum   += lambda_squared
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

        reconstructed_temp = th.lerp(
            self.temp_min, self.temp_max, displayed_rgb[..., 0]
        )

        if (n_temps := sensed_temperature.numel()) > 0:
            self.sum   += (sensed_temperature - reconstructed_temp).abs().sum()
            self.count += n_temps


class DynamicBalanceMetric(AveragingMetric):
    """
    Measures balance between expansion and contraction in alert mode.

    Tracks the ratio of density changes to ensure the flock maintains
    appropriate compactness during threat response without collapse.

    The balance ratio is:
        β = r_avg / d_avg

    where r_avg is average distance to center and d_avg is average pairwise distance.
    """
    
    def update(
        self,
        in_alert_mode : bool,
        positions     : Tensor
    ):
        """
        Update metric with current density ratio.

        Args:
            in_alert_mode : Whether flock is in alert mode
            positions     : Tensor [N, 3] of agent positions
        """
        if not in_alert_mode or len(positions) < 2:
            return

        center     = positions.mean(dim=0)
        avg_radius = (positions - center).norm(dim=1).mean()
        pairwise   = th.cdist(positions, positions).triu(1)
        
        if (n_pairs := pairwise.gt(0).sum()) > 0:
            avg_dist = pairwise.sum() / n_pairs
            self.sum   += avg_radius / avg_dist.clamp_min(1e-8)
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

        thrust_magnitude = (u_safe - gravity_vector).norm(dim=-1)

        self.sum   += thrust_magnitude.pow(self.power_exponent).sum()
        self.count += u_safe.shape[0]


class InformationPropagationMetric(AveragingMetric):
    """
    Measures information propagation speed through the flock.

    From Cavagna et al. (2010), information propagates at:
        v_info = c_0 √(χ/m_eff) ∈ [15, 45] m/s

    This metric estimates propagation speed by tracking velocity changes
    from edge to center of the flock.
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
    
    def update(
        self,
        positions  : Tensor,
        velocities : Tensor
    ):
        """
        Update metric with velocity change propagation estimate.

        Args:
            positions  : Tensor [N, 3] of agent positions
            velocities : Tensor [N, 3] of agent velocities
        """
        if self.previous_velocities is None:
            self.previous_velocities = velocities.clone()
            return

        v_changes = (velocities - self.previous_velocities).norm(dim=1)
        
        if (threshold := v_changes.mean() + v_changes.std()) > 0 and \
           (significant := v_changes > threshold).any():
            center      = positions.mean(dim=0)
            radii       = (positions - center).norm(dim=1)
            mean_radius = radii.mean()
            
            if (edge_agents := significant & (radii > mean_radius)).any():
                propagation_speed = (
                    (radii.max() - radii.min()) / self.time_step
                )
                
                normalized = (
                    (propagation_speed - self.target_min) / 
                    (self.target_max   - self.target_min)
                ).clamp(0, 1)
                
                self.sum   += normalized
                self.count += 1
        
        self.previous_velocities = velocities.clone()


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
        Pre-compute 2D coordinate grid for KDE rendering.

        Creates a meshgrid of (x, y) coordinates used for evaluating
        Gaussian kernels during velocity field rendering.

        Args:
            grid_size : Resolution of the rendering grid

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

        distance_squared = (
            (self.coords.unsqueeze(2) - grid_positions) ** 2
        ).sum(dim=-1)
        
        kernel_weights = th.exp(-distance_squared / (2 * self.sigma ** 2))
        field          = (kernel_weights * velocity_magnitude).sum(dim=-1)

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
            positions  : Agent positions 𝐱 ∈ ℝ^(n×3)
            velocities : Agent velocities 𝐯 ∈ ℝ^(n×3)  
            wind       : Environmental wind field 𝐰 ∈ ℝ^(n×3)
        """
        flock_field = self._render_velocity_field(
            bounds_max = self.bounds_max,
            bounds_min = self.bounds_min,
            positions  = positions,
            velocities = velocities
        )

        wind_field = self._render_velocity_field(
            bounds_max = self.bounds_max,
            bounds_min = self.bounds_min,
            positions  = positions,
            velocities = wind
        )

        ssim_value = self.ssim_metric(
            preds  = flock_field.unsqueeze(0).unsqueeze(0),
            target = wind_field.unsqueeze(0).unsqueeze(0)
        )

        self.sum   += ssim_value
        self.count += 1


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
        spins      = th.nn.functional.normalize(velocities, dim=1)
        delta_spin = spins - spins.mean(dim=0, keepdim=True)
        
        distances = th.cdist(positions, positions)
        corr_mat  = delta_spin @ delta_spin.T
        
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
        
        return -(X @ Y) / (X @ X) if (X @ X) > 0 else 0

    def update(
        self,
        positions  : Tensor,
        velocities : Tensor
    ):
        """
        Update metric with scale-free correlation measurement.
        
        Computes velocity correlations, bins by distance, and fits
        a power law to measure deviation from expected exponent.
        
        Args:
            positions  : Agent positions 𝐱 ∈ ℝ^(n×3)
            velocities : Agent velocities 𝐯 ∈ ℝ^(n×3)
        """
        corr_mat, distances = self._compute_velocity_correlations(
            positions, velocities
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
            
            self.sum   += (fitted_exponent - self.target_exponent).abs()
            self.count += 1


class SusceptibilityMetric(AveragingMetric):
    """
    Tracks flock susceptibility χ = N · Var[Φ].
    
    Monitors the system's responsiveness to perturbations, which should
    remain high (5-20) to maintain critical state dynamics.
    """
    
    def __init__(self, metrics: MetricsModel):
        """
        Initialize with target susceptibility range.
        
        Args:
            metrics : Metrics configuration containing susceptibility range
        """
        super().__init__()
        self.target_min = metrics.susceptibility_min
        self.target_max = metrics.susceptibility_max
    
    def update(
        self,
        velocities : Tensor
    ):
        """
        Update metric with current susceptibility measurement.
        
        Args:
            velocities : Tensor [N, 3] of agent velocities
        """
        normalized_vels = (
            velocities / 
            velocities.norm(dim=1, keepdim=True).clamp(min=1e-8)
        )
        
        if (n := len(velocities)) > 0:
            susceptibility = n * normalized_vels.var(dim=0).sum()
            in_range = (
                (self.target_min <= susceptibility) & 
                (susceptibility  <= self.target_max)
            )
            self.sum   += in_range.float()
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
            mmm : Murmuration model containing k_neighbors
        """
        super().__init__()
        self.k_neighbors        = mmm.k_neighbors
        self.previous_neighbors = None
    
    def update(
        self,
        positions : Tensor
    ):
        """
        Update metric with neighbor consistency measurement.
        
        Args:
            positions : Tensor [N, 3] of agent positions
        """
        if (n_agents := len(positions)) < self.k_neighbors + 1:
            return
            
        _, indices = th.cdist(positions, positions).topk(
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
        mmm        : MurmurationModel
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
        """
        self.bounds_max = bounds_max
        self.gravity    = gravity
        self.metrics    = metrics
        self.mmm        = mmm

        self._init_evaluation_metrics()
        self._init_imitation_metrics()
        self._init_murmuration_metrics()

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
    
    def _init_murmuration_metrics(self):
        """
        Initialize murmuration-specific evaluation metrics.
        
        These metrics assess the emergent properties of the murmuration dynamics:
        - correlation_mse      : Scale-free velocity correlations C(r) ~ r^(-1/3)
        - dynamic_balance      : Balance between expansion/contraction in alert mode
        - info_speed           : Information propagation speed (15-45 m/s target)
        - susceptibility       : System responsiveness χ = N·Var[Φ] (5-20 target)
        - topological_fidelity : Consistency of k-nearest neighbor connections
        
        Each metric is wrapped in MetricCollection for automatic train/val splitting.
        """
        self.train_murmuration = MetricCollection({
            "correlation_mse"      : ScaleFreeCorrelationMetric(self.mmm),
            "dynamic_balance"      : DynamicBalanceMetric(), 
            "info_speed"           : InformationPropagationMetric(self.metrics),
            "susceptibility"       : SusceptibilityMetric(self.metrics),
            "topological_fidelity" : TopologicalFidelityMetric(self.mmm)
        })
        self.val_murmuration = self.train_murmuration.clone(prefix="val_")

    def log_all_metrics(
        self,
        is_training : bool,
        loss        : Tensor | None,
        module      : LightningModule,
        predictions : Tensor | None,
        step_output : bool,
        targets     : Tensor | None
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
            loss        : Behavioral cloning loss (required if step_output=True)
            module      : Lightning module providing the logger interface
            predictions : Model velocity predictions 𝐯_pred ∈ ℝ^(n×3) [m/s]
            step_output : Whether this is step-level or epoch-end logging
            targets     : Expert velocity commands 𝐯_expert ∈ ℝ^(n×3) [m/s]
        """
        phase = "train" if is_training else "val"

        if step_output:
            assert loss        is not None, "Loss required for step logging"
            assert predictions is not None, "Predictions required for step logging"
            assert targets     is not None, "Targets required for step logging"

            module.log(
                name     = f"{phase}/loss",
                on_epoch = True,
                on_step  = is_training,
                prog_bar = True,
                value    = loss
            )

            for i, dim in enumerate(["x", "y", "z"]):
                module.log(
                    name     = f"{phase}/velocity_{dim}_mse",
                    on_epoch = True,
                    on_step  = False,
                    value    = (predictions[..., i] - targets[..., i]).pow(2).mean()
                )

        for metric_type, on_step in [
            ("imitation",   is_training),
            ("evaluation",  False),
            ("murmuration", False)
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
            metrics["mae_color"].update(
                displayed_rgb      = None,
                sensed_temperature = batch["temperature"]
            )

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

        if u_control := (batch.get("u_safe") or batch.get("action")):
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
    
    def update_murmuration_metrics(
        self,
        batch       : TensorDictBase,
        is_training : bool
    ):
        """
        Update murmuration-specific metrics from simulation state.
        
        Tracks emergent properties of the murmuration dynamics including
        scale-free correlations, critical state indicators, and topological
        consistency. These metrics verify the biological plausibility of
        the learned flocking behavior.
        
        Field requirements by metric:
        - correlation_mse      : position, velocity
        - dynamic_balance      : position, in_alert_mode flag
        - info_speed           : position, velocity (tracked over time)
        - susceptibility       : velocity
        - topological_fidelity : position (tracked over time)
        
        Args:
            batch       : TensorDict containing simulation state
            is_training : Whether this is training (True) or validation (False)
        """
        metrics = self._get_metrics("murmuration", is_training)
        
        if all(k in batch for k in ["position", "velocity"]):
            metrics["correlation_mse"].update(
                positions  = batch["position"],
                velocities = batch["velocity"]
            )
            
            metrics["info_speed"].update(
                positions  = batch["position"],
                velocities = batch["velocity"]
            )
        
        if "position" in batch and "in_alert_mode" in batch:
            metrics["dynamic_balance"].update(
                in_alert_mode = batch["in_alert_mode"],
                positions     = batch["position"]
            )
        
        if "velocity" in batch:
            metrics["susceptibility"].update(velocities = batch["velocity"])
        
        if "position" in batch:
            metrics["topological_fidelity"].update(positions = batch["position"])