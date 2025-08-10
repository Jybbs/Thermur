"""
Unified metrics collection for imitation learning training and evaluation.

This module provides a centralized MetricsCollector that manages all metrics
for the training pipeline, including imitation learning losses, core evaluation
metrics, and runtime performance tracking. The collector integrates seamlessly
with PyTorch Lightning's logging system and Weights & Biases.
"""
from __future__         import annotations
from collections        import Counter
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
    
    def _ensure_correct_device(self, reference_tensor: Tensor):
        """
        Ensure state variables are on the same device as the reference tensor.
        
        This is needed because TorchMetrics initializes states on CPU by default,
        but computations may happen on GPU/MPS.
        
        Args:
            reference_tensor: A tensor on the target device
        """
        if self.sum.device != reference_tensor.device:
            self.sum   = self.sum.to(reference_tensor.device)
            self.count = self.count.to(reference_tensor.device)

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

    def _compute_fiedler_power_iteration(
        self,
        laplacian  : Tensor,
        iterations : int = 30
    ) -> Tensor:
        """
        Compute Fiedler value using power iteration method.
        
        MPS-compatible alternative to eigvalsh that avoids CPU fallback.
        Uses inverse power iteration with shift to find second smallest eigenvalue.
        
        Args:
            laplacian  : Graph Laplacian matrix [n, n]
            iterations : Number of power iterations
            
        Returns:
            Approximation of second smallest eigenvalue (Fiedler value)
        """
        n      = laplacian.shape[0]
        device = laplacian.device
        
        if n <= 1:
            return th.tensor(0.0, device=device)
        
        random_vector = th.randn(n, device=device)
        orthogonal_vector = random_vector - random_vector.mean()
        v = orthogonal_vector / orthogonal_vector.norm()
        
        shift = 0.001
        shifted_laplacian = laplacian + shift * th.eye(n, device=device)
        
        for _ in range(iterations):
            v_new = th.linalg.solve(shifted_laplacian, v)
            orthogonalized = v_new - v_new.mean()
            
            if (norm := orthogonalized.norm()) > 1e-10:
                v = orthogonalized / norm
            else:
                break
        
        rayleigh_numerator   = v @ (laplacian @ v)
        rayleigh_denominator = v @ v
        fiedler_value        = rayleigh_numerator / rayleigh_denominator
        
        return fiedler_value.clamp_min(0.0)
    
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
        adjacency_matrix = th.zeros(
            (num_agents, num_agents), device=edge_index.device
        )
        
        if edge_index.numel() > 0:
            adjacency_matrix[edge_index[0], edge_index[1]] = 1.0
            adjacency_matrix[edge_index[1], edge_index[0]] = 1.0
        
        degree_matrix = th.diag_embed(adjacency_matrix.sum(1))
        laplacian     = degree_matrix - adjacency_matrix
        
        return laplacian

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

        fiedler_value = self._compute_fiedler_power_iteration(laplacian)
        
        self._ensure_correct_device(fiedler_value)
        self.sum   += fiedler_value
        self.count += 1


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
            balance_ratio = (
                (batch["position"] - center).norm(dim=1).mean() / 
                (pairwise.sum() / n_pairs).clamp_min(1e-8)
            )
            self._ensure_correct_device(balance_ratio)
            self.sum   += balance_ratio
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

        power_sum = thrust_magnitude.pow(self.power_exponent).sum()
        self._ensure_correct_device(power_sum)
        self.sum   += power_sum
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
        device     = positions.device
        bounds_max = bounds_max.to(device)
        bounds_min = bounds_min.to(device)
        coords     = self.coords.to(device)
        
        velocity_magnitude = velocities[:, :2].norm(dim=1)
        grid_positions     = (
            (positions[:, :2] - bounds_min[:2]) /
            (bounds_max[:2]   - bounds_min[:2]) *
            (self.grid_size - 1)
        )
        
        sparse_condition = (
            self.grid_size > 32 and 
            positions.shape[0] < self.grid_size * 2
        )
        
        if sparse_condition:
            field = th.zeros(
                (self.grid_size, self.grid_size), device=device
            )
            kernel_radius = int(3 * self.sigma)
            
            for idx, grid_pos in enumerate(grid_positions):
                x_min = max(0, int(grid_pos[0] - kernel_radius))
                x_max = min(self.grid_size, int(grid_pos[0] + kernel_radius + 1))
                y_min = max(0, int(grid_pos[1] - kernel_radius))
                y_max = min(self.grid_size, int(grid_pos[1] + kernel_radius + 1))
                
                if x_max <= x_min or y_max <= y_min:
                    continue
                
                local_region      = coords[y_min:y_max, x_min:x_max]
                height, width     = local_region.shape[:2]
                local_region_flat = local_region.reshape(-1, 2)
                
                # Compute Gaussian kernel weights
                squared_distances = (
                    (local_region_flat - grid_pos.unsqueeze(0)) ** 2
                ).sum(dim=1)
                
                kernel_weights = th.exp(
                    -squared_distances / (2 * self.sigma ** 2)
                )
                
                field[y_min:y_max, x_min:x_max] += (
                    kernel_weights.view(height, width) * velocity_magnitude[idx]
                )
        else:
            flattened_coords  = coords.view(-1, 2)
            squared_distances = th.cdist(flattened_coords, grid_positions).pow(2)
            kernel_weights    = th.exp(-squared_distances / (2 * self.sigma ** 2))
            weighted_field    = (kernel_weights * velocity_magnitude).sum(dim=-1)
            field             = weighted_field.view(self.grid_size, self.grid_size)

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
        batch_size   = positions.shape[0]
        device       = positions.device
        field_shape  = (batch_size, 1, self.grid_size, self.grid_size)
        flock_fields = th.zeros(field_shape, device=device)
        wind_fields  = th.zeros(field_shape, device=device)
        
        for i in range(batch_size):
            flock_fields[i, 0] = self._render_velocity_field(
                bounds_max = self.bounds_max,
                bounds_min = self.bounds_min,
                positions  = positions[i],
                velocities = velocities[i]
            )
            
            wind_fields[i, 0] = self._render_velocity_field(
                bounds_max = self.bounds_max,
                bounds_min = self.bounds_min,
                positions  = positions[i],
                velocities = wind[i]
            )
        
        self.ssim_metric = self.ssim_metric.to(device)
        ssim_values = self.ssim_metric(
            preds=flock_fields, target=wind_fields
        )
        
        total_ssim = ssim_values.sum()
        self._ensure_correct_device(total_ssim)
        self.sum   += total_ssim
        self.count += batch_size


class StateMetrics(Metric):
    """
    Tracks average physical state properties of the flock.
    
    Monitors key physical quantities including velocity magnitude |𝐯|,
    temperature θ, and acceleration magnitude |𝐚| across all agents.
    These reveal whether the learned policy maintains realistic flight
    dynamics matching both the expert controller and empirical observations
    from Cavagna et al. (2010) and Attanasi et al. (2014).
    
    The metrics track:
        - |𝐯|_avg : Mean velocity magnitude, expected 10-20 m/s in cruise
                    (starlings typically fly at 15 m/s per Cavagna et al.)
        - θ_avg   : Mean sensed temperature across flock, critical for
                    thermal safety constraints
        - |𝐚|_avg : Mean acceleration magnitude, indicating control effort
                    and energy expenditure (typical range 5-15 m/s²)
    
    These quantities help diagnose whether the learned policy captures the
    active matter dynamics of self-propelled particles maintaining constant
    speed while adapting to environmental gradients.
    """
    acceleration_sum : Tensor
    count            : Tensor
    temperature_sum  : Tensor
    velocity_sum     : Tensor
    
    def __init__(self):
        """
        Initialize state tracking for physical quantities.
        """
        super().__init__()
        self.add_state("acceleration_sum", th.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count",            th.tensor(0),   dist_reduce_fx="sum")
        self.add_state("temperature_sum",  th.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("velocity_sum",     th.tensor(0.0), dist_reduce_fx="sum")
    
    def compute(self) -> dict[str, Tensor]:
        """
        Compute averages from accumulated sums.
        
        Returns:
            Dictionary containing avg_acceleration, avg_temperature, avg_velocity
        """
        if self.count == 0:
            zero = th.tensor(0.0)
            return {
                "avg_acceleration" : zero,
                "avg_temperature"  : zero,
                "avg_velocity"     : zero,
            }
        
        count = self.count.float()
        return {
            "avg_acceleration" : self.acceleration_sum / count,
            "avg_temperature"  : self.temperature_sum  / count,
            "avg_velocity"     : self.velocity_sum     / count,
        }
    
    def update(self, batch: TensorDictBase):
        """
        Update running sums with batch statistics.
        
        Extracts physical quantities from the batch and accumulates their
        magnitudes for computing running averages.
        
        Args:
            batch: TensorDict containing velocity, temperature, and optionally action
        """
        if "action" in batch:
            action = batch["action"]
            if action.dim() == 3:
                action = action.reshape(-1, action.shape[-1])
            self.acceleration_sum += action.norm(dim=-1).mean()
        
        if "temperature" in batch and batch["temperature"].numel() > 0:
            self.temperature_sum += batch["temperature"].mean()
        
        if "velocity" in batch:
            velocity = batch["velocity"]
            if velocity.dim() == 3:
                velocity = velocity.reshape(-1, velocity.shape[-1])
            self.velocity_sum += velocity.norm(dim=-1).mean()
        
        self.count += 1


class ConnectivityMetrics(Metric):
    """
    Tracks topological connectivity of the k-nearest neighbor graph.
    
    Monitors how well the topological interaction structure is maintained
    during flight dynamics. Ballerini et al. (2008) discovered that starlings
    interact with a fixed number of k = 6-7 nearest neighbors regardless of
    metric distance, a topological rule that enables scale-free correlations
    and optimal information transfer (Bialek et al., 2012).
    
    The metrics track:
        - k_avg : Mean degree ⟨k_i⟩ across the flock, should stabilize near 7
        - ρ_k   : Fraction of agents maintaining k ≥ k_target neighbors,
                  indicates structural integrity of the interaction network
        - N_iso : Count of isolated agents with k < 3 (danger threshold),
                  as agents with fewer than 3 neighbors cannot triangulate
                  information and lose flock cohesion
    
    where k_i = |𝒩_i| is the neighbor count for agent i. The topological
    interaction rule is critical for achieving the maximum entropy state
    that balances individual freedom with collective response.
    """
    connected_ratio_sum : Tensor
    count               : Tensor
    isolated_sum        : Tensor
    k_target            : int
    neighbors_sum       : Tensor
    
    def __init__(self, k_target: int = 7):
        """
        Initialize connectivity tracking with target degree.
        
        Args:
            k_target: Target number of neighbors (default 7 from Ballerini, 2008)
        """
        super().__init__()
        self.k_target = k_target
        self.add_state("connected_ratio_sum", th.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count",               th.tensor(0),   dist_reduce_fx="sum")
        self.add_state("isolated_sum",        th.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("neighbors_sum",       th.tensor(0.0), dist_reduce_fx="sum")
    
    def compute(self) -> dict[str, Tensor]:
        """
        Compute connectivity statistics from accumulated data.
        
        Returns:
            Dictionary containing avg_neighbors, connectivity_ratio, isolated_agents
        """
        if self.count == 0:
            zero = th.tensor(0.0)
            return {
                "avg_neighbors"      : zero,
                "connectivity_ratio" : zero,
                "isolated_agents"    : zero,
            }
        
        count = self.count.float()
        return {
            "avg_neighbors"      : self.neighbors_sum / count,
            "connectivity_ratio" : self.connected_ratio_sum / count,
            "isolated_agents"    : self.isolated_sum / count,
        }
    
    def update(self, batch: TensorDictBase):
        """
        Update connectivity statistics from graph topology.
        
        Computes degree distribution from the edge list and tracks how well
        the k-NN structure is maintained under dynamics.
        
        Args:
            batch: TensorDict containing edge_index and position tensors
        """
        if "edge_index" not in batch or "position" not in batch:
            return
        
        edge_index = batch["edge_index"]
        if edge_index.dim() == 3:
            edge_index = edge_index[0]
        
        if edge_index.numel() == 0:
            return
        
        n_agents = (
            batch["position"].shape[1] if batch["position"].dim() == 3
            else batch["position"].shape[0]
        )
        
        neighbor_counts = th.bincount(
            input     = edge_index[0].long(),
            minlength = n_agents
        ).float()
        
        self.connected_ratio_sum += (neighbor_counts >= self.k_target).float().mean()
        self.isolated_sum        += (neighbor_counts < 3).float().sum()
        self.neighbors_sum       += neighbor_counts.mean()
        self.count               += 1


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
        Compute metrics that have received sufficient data.
        
        Returns computed metrics only when all have been properly updated.
        This prevents warnings during sanity checks when compute() is called
        before update() methods.
        
        Readiness criteria by metric type:
        - Custom metrics (AveragingMetric, StateMetrics, etc.): count > 0
        - R2Score: Requires at least 2 updates to compute correlation
        - Standard metrics (MAE, MSE): At least 1 update required
        - Unknown types: Check for _update_count attribute > 0
        
        Args:
            metrics: Collection of metrics to potentially compute
            
        Returns:
            Dictionary of computed values or None if any metric lacks data
        """
        readiness = Counter()
        
        for metric in metrics.values():
            if isinstance(
                metric, 
                (
                    AveragingMetric,      ConnectivityMetrics,
                    DynamicBalanceMetric, StateMetrics 
                )
            ):
                is_ready = hasattr(metric, 'count') and metric.count > 0
            elif isinstance(metric, R2Score):
                is_ready = getattr(metric, '_update_count', 0) >= 2
            elif isinstance(metric, (MeanAbsoluteError, MeanSquaredError)):
                is_ready = getattr(metric, '_update_count', 0) > 0
            else:
                is_ready = getattr(metric, '_update_count', 0) > 0
            
            readiness[is_ready] += 1
        
        if readiness[False] == 0 and readiness[True] > 0:
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
            "avg_power" : EnergyConsumptionMetric(self.gravity, self.metrics),
            "λ₂"        : CohesionMetric(),
            "ssim"      : LegibilitySSIMMetric(self.bounds_max, self.metrics),
        })
        self.val_evaluation = self.train_evaluation.clone()

    def _init_imitation_metrics(self):
        """
        Initialize metrics for evaluating imitation learning performance.

        These metrics include regression accuracy and spatial murmuration properties:
        - dynamic_balance  : Density ratio under thermal threat
        - MSE/RMSE/MAE/R²  : Regression metrics for velocity predictions
        - scale_free_error : Deviation from power-law correlations C(r) ~ r^(-1/3)

        All metrics work on individual frames without temporal dependencies.
        """
        create_imitation_metrics = lambda prefix="": MetricCollection(
            {
                "connectivity"    : ConnectivityMetrics(self.mmm.k_neighbors),
                "dynamic_balance" : DynamicBalanceMetric(self.safety),
                "mae"             : MeanAbsoluteError(),
                "mse"             : MeanSquaredError(),
                "r2"              : R2Score(multioutput='uniform_average'),
                "rmse"            : MeanSquaredError(squared=False),
                "scale_free"      : ScaleFreeCorrelationMetric(self.mmm),
                "state"           : StateMetrics()
            }, 
            compute_groups = False,
            prefix         = prefix)
        
        self.train_imitation = create_imitation_metrics()
        self.val_imitation   = create_imitation_metrics()
    

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
                name      = f"{phase}/loss",
                on_epoch  = True,
                on_step   = is_training,
                prog_bar  = True,
                sync_dist = True,
                value     = step_data["loss"]
            )

            for i, dim in enumerate(["x", "y", "z"]):
                module.log(
                    name      = f"{phase}/velocity_{dim}_mse",
                    on_epoch  = True,
                    on_step   = False,
                    sync_dist = True,
                    value     = (step_data["predictions"][..., i] - 
                                step_data["targets"][..., i]).pow(2).mean()
                )
            
            if computed := self._compute_ready_metrics(
                self._get_metrics(is_training, "imitation")
            ):
                prefixed = {f"{phase}/{k}": v for k, v in computed.items()}
                module.log_dict(
                    dictionary = prefixed,
                    on_epoch   = True,
                    on_step    = is_training,
                    sync_dist  = True
                )
        
        if not is_training or step_data is None:
            if computed := self._compute_ready_metrics(
                self._get_metrics(is_training, "evaluation")
            ):
                prefixed = {f"{phase}/{k}": v for k, v in computed.items()}
                module.log_dict(
                    dictionary = prefixed,
                    on_epoch   = True,
                    on_step    = is_training,
                    sync_dist  = True,
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
            if name in metrics:
                metrics[name].update(predictions, targets)
        
        if batch is not None:
            for name in ["connectivity", "dynamic_balance", "scale_free", "state"]:
                if name in metrics:
                    metrics[name].update(batch)
    
