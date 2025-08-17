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
    from config.imitation.training   import MetricsModel
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
        self.add_state("count", th.tensor(0),   "sum")
        self.add_state("sum",   th.tensor(0.0), "sum")
    
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
        
        random_vector     = th.randn(n, device=device)
        orthogonal_vector = random_vector - random_vector.mean()
        v                 = orthogonal_vector / orthogonal_vector.norm()
        
        shift = 0.001
        shifted_laplacian = laplacian + shift * th.eye(n, device=device)
        
        for _ in range(iterations):
            v_new          = th.linalg.solve(shifted_laplacian, v)
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
            batch_size     = edge_index.shape[0]
            fiedler_values = th.zeros(batch_size, device=edge_index.device)
            
            for i in range(batch_size):
                if edge_index[i].numel() > 0:
                    laplacian = self._compute_graph_laplacian(edge_index[i], num_agents)
                    fiedler_values[i] = self._compute_fiedler_power_iteration(laplacian)
            
            self.sum   += fiedler_values.sum()
            self.count += batch_size
            return

        laplacian     = self._compute_graph_laplacian(edge_index, num_agents)
        fiedler_value = self._compute_fiedler_power_iteration(laplacian)
        
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
        self.sum   += power_sum
        self.count += u_safe.shape[0]


class HamiltonianEnergyMetric(AveragingMetric):
    """
    Track Hamiltonian energy E = -Σ J_ij s_i·s_j per timestep.

    Computes the interaction energy of the flock using a physics-inspired
    Hamiltonian model where agents are treated as spins with pairwise
    coupling that decays exponentially with distance.
    """

    def __init__(self, mmm: MurmurationModel):
        """
        Initialize with murmuration model parameters.

        Args:
            mmm: Murmuration model containing coupling parameters
        """
        super().__init__()
        self.j_base                = mmm.j_base
        self.coupling_decay        = mmm.coupling_decay
        self.alert_coupling_factor = mmm.alert_coupling_factor

    def update(self, batch: TensorDictBase):
        """
        Compute Hamiltonian energy matching controller implementation.
        
        Implements the exact energy formulation from MurmurationController:
        E = -Σ_{<ij>} J_{ij}^{alert} 𝐬_i · 𝐬_j
        
        where J_{ij}^{alert} = κ_i × J_0 exp(-d_{ij}/λ) with:
        - κ_i = 1.0 for relaxed birds, alert_coupling_factor for alert birds  
        - d_{ij} is topological distance from k-NN graph
        - Only includes edges that exist in the controller's graph

        Args:
            batch: TensorDict containing velocity, alert_states, edge indices, 
                   and topo_distances from controller
        """
        spins = th.nn.functional.normalize(batch["velocity"], dim=-1)
        
        if "topo_distances" not in batch or "edge_source" not in batch:
            distances = th.cdist(batch["position"], batch["position"])
            coupling  = self.j_base * th.exp(-distances / self.coupling_decay)
            coupling.diagonal(dim1=-2, dim2=-1).fill_(0)
        else:
            batch_size, n_agents = spins.shape[:2]
            coupling  = th.zeros(
                batch_size, 
                n_agents, 
                n_agents, 
                device = spins.device
            )
            
            if batch["edge_source"].numel() > 0:
                n_edges       = batch["edge_source"].shape[1]
                batch_indices = th.arange(
                    device = spins.device,
                    end    = batch_size
                ).unsqueeze(1).expand(-1, n_edges)
                
                alert_states_source = batch["alert_states"][
                    batch_indices, batch["edge_source"]
                ]
                coupling_modifier = th.where(
                    alert_states_source > 0.5,
                    self.alert_coupling_factor,
                    1.0
                )
                
                j_edges = self.j_base * coupling_modifier * th.exp(
                    -batch["topo_distances"][
                        batch_indices, batch["edge_source"], batch["edge_target"]
                    ] / self.coupling_decay
                )
                
                coupling[batch_indices, batch["edge_source"], 
                        batch["edge_target"]] = j_edges
                coupling[batch_indices, batch["edge_target"], 
                        batch["edge_source"]] = j_edges

        spin_products = spins @ spins.transpose(-2, -1)
        energies      = -(coupling * spin_products).sum(dim=(-2, -1)) / 2
        
        self.sum   += energies.sum()
        self.count += energies.numel()


class InformationPropagationMetric(AveragingMetric):
    """
    Measures empirical information propagation speed through the flock.
    
    Tracks how velocity perturbations propagate from edge to center by analyzing
    the spatial decay of velocity correlations. The correlation function follows:
    
        C(r) = ⟨𝐯̂ᵢ · 𝐯̂ⱼ⟩ ~ exp(-r/ξ)
    
    where ξ is the correlation length. Information speed is estimated as:
    
        v_info = ξ / Δt
    
    Natural murmurations exhibit v_info ∈ [15, 45] m/s (Attanasi et al. 2014),
    with higher speeds indicating more responsive collective dynamics. The range
    depends on flock alertness:
        - Relaxed state: v_info ≈ 15 m/s (low responsiveness)  
        - Alert state: v_info ≈ 45 m/s (high responsiveness)
    
    This metric reveals whether the learned policy maintains proper information
    transfer rates matching empirical observations.
    """
    
    def __init__(self, metrics: MetricsModel):
        """
        Initialize with target propagation speed parameters.
        
        Args:
            metrics: Configuration containing info_propagation_* parameters
        """
        super().__init__()
        self.target_max = metrics.info_propagation_max_speed
        self.target_min = metrics.info_propagation_min_speed
        self.time_step  = metrics.info_propagation_time_step
    
    def update(self, batch: TensorDictBase):
        """
        Estimate propagation speed from velocity correlation decay.
        
        Args:
            batch: TensorDict containing position and velocity tensors
        """
        position = (
            batch["position"] if batch["position"].dim() == 3 
            else batch["position"].unsqueeze(0)
        )
        velocity = (
            batch["velocity"] if batch["velocity"].dim() == 3
            else batch["velocity"].unsqueeze(0)
        )
        
        n_agents = position.shape[-2]
        if n_agents < 10:
            return
        
        distances    = th.cdist(position, position)
        normed_vel   = th.nn.functional.normalize(velocity, dim=-1)
        correlations = th.bmm(normed_vel, normed_vel.transpose(-2, -1)).abs()
        mask         = ~th.eye(n_agents, device=position.device, dtype=th.bool)
        high_corr    = (correlations > 0.5) & mask
        
        speeds = th.where(
            high_corr,
            distances / self.time_step,
            th.tensor(0.0, device=distances.device)
        )[high_corr].clamp(self.target_min, self.target_max)
        
        if speeds.numel() > 0:
            self.sum   += speeds.mean() * position.shape[0]
            self.count += position.shape[0]


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
        coords     = self.coords
        bounds_min = self.bounds_min
        bounds_max = self.bounds_max
        
        velocity_magnitude = velocities[:, :2].norm(dim=1)
        grid_positions     = (
            (positions[:, :2] - bounds_min[:2]) /
            (bounds_max[:2]   - bounds_min[:2]) *
            (self.grid_size - 1)
        )
        
        flattened_coords  = coords.view(-1, 2)
        squared_distances = th.cdist(flattened_coords, grid_positions).pow(2)
        kernel_weights    = th.exp(-squared_distances / (2 * self.sigma ** 2))
        field             = (kernel_weights @ velocity_magnitude).view(self.grid_size, self.grid_size)

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
        
        ssim_values = self.ssim_metric(
            preds=flock_fields, target=wind_fields
        )
        
        total_ssim = ssim_values.sum()
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
        self.add_state("acceleration_sum", th.tensor(0.0), "sum")
        self.add_state("count",            th.tensor(0),   "sum")
        self.add_state("temperature_sum",  th.tensor(0.0), "sum")
        self.add_state("velocity_sum",     th.tensor(0.0), "sum")
    
    def _flatten_agent_batch(self, tensor: Tensor) -> tuple[Tensor, int]:
        """
        Flatten hierarchical agent batches for statistical computation.
        
        Transforms multi-agent batch tensors from [batch, agents, features]
        format to [batch*agents, features] format for computing per-agent
        statistics across the entire flock.
        
        Args:
            tensor: Input with shape [batch, agents, features] for batched
                    trajectories, [agents, features] for single timesteps,
                    or [features] for single agents
        
        Returns:
            Tuple of (flattened_tensor, n_samples) where flattened has shape
            [total_samples, features] and n_samples counts individual agents
        """
        if tensor.dim() == 3:
            shape = tensor.shape
            return tensor.reshape(-1, shape[2]), shape[0] * shape[1]
        
        elif tensor.dim() == 2:
            return tensor, tensor.shape[0]
        
        else:
            return tensor.unsqueeze(0), 1
    
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
        magnitudes for computing running averages across all agents.
        Only tracks states when control actions are present.
        
        Args:
            batch: TensorDict containing velocity, temperature, and optionally action
        """
        if "action" not in batch:
            return
        
        flattened_action, n_samples = self._flatten_agent_batch(batch["action"])
        self.acceleration_sum      += flattened_action.norm(dim=-1).sum()
        self.count                 += n_samples
        
        if "temperature" in batch and batch["temperature"].numel() > 0:
            self.temperature_sum += batch["temperature"].sum()
        
        if "velocity" in batch:
            flattened_velocity, _ = self._flatten_agent_batch(batch["velocity"])
            self.velocity_sum    += flattened_velocity.norm(dim=-1).sum()


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
        self.add_state("connected_ratio_sum", th.tensor(0.0), "sum")
        self.add_state("count",               th.tensor(0),   "sum")
        self.add_state("isolated_sum",        th.tensor(0.0), "sum")
        self.add_state("neighbors_sum",       th.tensor(0.0), "sum")
    
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
    
    def __init__(self, metrics: MetricsModel):
        """
        Initialize with target correlation exponent.
        
        Args:
            metrics: Metrics model with expected exponent γ
        """
        super().__init__()
        self.target_exponent = metrics.correlation_exponent
    
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
        
        Computes velocity correlations C(r) as a function of distance r and fits
        power law C(r) ~ r^(-γ) to measure deviation from expected scaling. Uses
        adaptive logarithmic binning with n_bins ∈ [3, 10] based on flock size:
        
            n_bins = min(10, max(3, n_pairs // 10))
        
        Handles edge cases where d_min = d_max (no variation) or n_agents < 4
        (insufficient data for power law fitting).
        
        Args:
            batch: TensorDict containing position and velocity
        """
        corr_mat, distances = self._compute_velocity_correlations(
            batch["position"], batch["velocity"]
        )
        
        triu_mask = th.triu(th.ones_like(distances), diagonal=1).bool()
        if not triu_mask.any():
            return
        
        unique_distances = distances[triu_mask]
        max_dist         = unique_distances.max()
        min_dist         = unique_distances.min()
        
        if min_dist == max_dist:
            return
        
        n_pairs = triu_mask.sum().item()
        n_bins  = min(10, max(3, n_pairs // 10))
        
        bin_edges = th.logspace(
            end   = max_dist.log10(),
            start = min_dist.log10(),
            steps = int(n_bins + 1)
        )
        
        bin_stats = [
            (corr_mat[mask].mean(), distances[mask].mean())
            for low, high in pairwise(bin_edges)
            if (mask := triu_mask & distances.gt(low) & distances.le(high)).any()
        ]
        
        if len(bin_stats) >= 3:
            bin_data        = th.tensor(bin_stats, device=distances.device)
            fitted_exponent = self._fit_power_law(
                bin_correlations = bin_data[:, 0],
                bin_distances    = bin_data[:, 1]
            )
            
            self.count += 1
            self.sum   += abs(fitted_exponent - self.target_exponent)


class SusceptibilityMetric(AveragingMetric):
    """
    Measures flock susceptibility to directional perturbations.
    
    Computes the normalized variance of the order parameter as a proxy for
    susceptibility to external stimuli. Following Bialek et al. (2012), the
    susceptibility quantifies collective response:
    
        χ = N · Var[Φ]
    
    where the order parameter Φ measures global alignment:
    
        Φ = |Σᵢ 𝐬ᵢ| / N
    
    with 𝐬ᵢ = 𝐯ᵢ/|𝐯ᵢ| being the normalized velocity (spin) of agent i.
    
    Natural murmurations maintain χ > 5, indicating proximity to a critical
    phase transition that maximizes:
        - Dynamic range (response to weak and strong signals)
        - Information capacity (bandwidth for signal propagation)
        - Correlation length (scale-free spatial correlations)
    
    Lower susceptibility (χ < 5) indicates an ordered state with reduced
    responsiveness, while very high susceptibility (χ > 20) suggests
    instability. The critical regime χ ∈ [5, 15] balances individual
    freedom with collective coordination.
    """
    
    def __init__(self, metrics: MetricsModel):
        """
        Initialize with susceptibility configuration.
        
        Args:
            metrics: Metrics configuration with susceptibility range
        """
        super().__init__()
        self.target_min = metrics.susceptibility_min
        self.target_max = metrics.susceptibility_max
    
    def update(self, batch: TensorDictBase):
        """
        Compute susceptibility from velocity fluctuations.
        
        Args:
            batch: TensorDict containing velocity tensor [n_agents, 3]
        """
        velocity = (
            batch["velocity"] if batch["velocity"].dim() == 3
            else batch["velocity"].unsqueeze(0)
        )
        
        n_agents = velocity.shape[-2]
        if n_agents < 2:
            return
        
        spins = th.nn.functional.normalize(velocity, dim=-1)
        mean_spin = spins.mean(dim=-2, keepdim=True)
        polarizations = (spins * mean_spin).sum(dim=-1)
        susceptibilities = n_agents * polarizations.var(dim=-1)
        
        self.sum   += susceptibilities.sum()
        self.count += susceptibilities.numel()


class TopologicalFidelityMetric(AveragingMetric):
    """
    Measures temporal stability of topological interaction networks.
    
    Quantifies how well the flock maintains k-nearest neighbor relationships
    over time using the Jaccard similarity coefficient between consecutive
    edge sets:
    
        J(E_t, E_{t+1}) = |E_t ∩ E_{t+1}| / |E_t ∪ E_{t+1}|
    
    where E_t = {(i,j) : j ∈ kNN(i)} is the edge set at time t.
    
    Ballerini et al. (2008) discovered that starlings interact with a fixed
    number k ≈ 7 topological neighbors regardless of metric distance. This
    topological rule enables:
        - Scale-free correlations C(r) ~ r^(-1/3)
        - Optimal information transfer with v_info ∈ [15, 45] m/s
        - Maximum entropy state balancing order and disorder
    
    High fidelity (J > 0.7) indicates stable neighborhoods that maintain
    information channels, while low fidelity (J < 0.3) suggests chaotic
    restructuring that disrupts collective response. The learned policy
    should maintain J ∈ [0.6, 0.8] for proper murmuration dynamics.
    """
    
    def __init__(self):
        """
        Initialize fidelity tracking with edge memory.
        """
        super().__init__()
        self.previous_edges: th.Tensor | None = None
    
    def update(self, batch: TensorDictBase):
        """
        Compute Jaccard similarity between consecutive edge sets.
        
        Args:
            batch: TensorDict containing edge_index [2, E] and trajectory_id
        """
        if "edge_index" not in batch:
            return
        
        edge_index = (
            batch["edge_index"][0] if batch["edge_index"].dim() == 3
            else batch["edge_index"]
        )
        
        if edge_index.shape[1] > 0:
            n_agents = edge_index.max().item() + 1
            edge_ids = edge_index[0] * n_agents + edge_index[1]
        else:
            edge_ids = th.empty(0, dtype=th.long, device=edge_index.device)
        
        if "trajectory_id" in batch:
            traj_ids     = batch["trajectory_id"] 
            current_traj = (
                traj_ids.flatten()[0].item() if traj_ids.numel() > 1
                else traj_ids.item()
            )
            if (
                hasattr(self, "last_trajectory_id") 
                and current_traj != self.last_trajectory_id
            ):
                self.previous_edges = None
            self.last_trajectory_id = current_traj
        
        if self.previous_edges is not None:
            curr_unique = th.unique(edge_ids)
            prev_unique = self.previous_edges
            
            if curr_unique.numel() > 0 and prev_unique.numel() > 0:
                
                # Intersection via mutual membership testing
                intersection = th.isin(curr_unique, prev_unique).sum()
                union = curr_unique.numel() + prev_unique.numel() - intersection
                
                if union > 0:
                    fidelity = intersection.float() / union
                    self.sum   += fidelity
                    self.count += 1
        
        self.previous_edges = th.unique(edge_ids) if edge_ids.numel() > 0 else None


class MetricsCollector(th.nn.Module):
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
        super().__init__()
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
            "avg_power"   : EnergyConsumptionMetric(self.gravity, self.metrics),
            "hamiltonian" : HamiltonianEnergyMetric(self.mmm),
            "λ₂"          : CohesionMetric(),
            "ssim"        : LegibilitySSIMMetric(self.bounds_max, self.metrics),
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
                "connectivity"      : ConnectivityMetrics(self.mmm.k_neighbors),
                "dynamic_balance"   : DynamicBalanceMetric(self.safety),
                "info_propagation"  : InformationPropagationMetric(self.metrics),
                "mae"               : MeanAbsoluteError(),
                "mse"               : MeanSquaredError(),
                "r2"                : R2Score(multioutput='uniform_average'),
                "rmse"              : MeanSquaredError(squared=False),
                "scale_free"        : ScaleFreeCorrelationMetric(self.metrics),
                "state"             : StateMetrics(),
                "susceptibility"    : SusceptibilityMetric(self.metrics),
                "topo_fidelity"     : TopologicalFidelityMetric()
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

        Uses a simplified logging strategy where training metrics are logged
        step-wise and validation metrics are logged epoch-wise. This eliminates
        the need for _step/_epoch suffixes since each metric logs at only one
        granularity.

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
        phase        = "training" if is_training else "validation"
        log_on_step  = is_training
        log_on_epoch = not is_training
        
        if step_data is not None:
            module.log(
                name      = f"{phase}/loss",
                on_epoch  = log_on_epoch,
                on_step   = log_on_step,
                prog_bar  = True,
                sync_dist = True,
                value     = step_data["loss"]
            )

            for i, dim in enumerate(["x", "y", "z"]):
                module.log(
                    name      = f"{phase}/velocity_{dim}_mse",
                    on_epoch  = log_on_epoch,
                    on_step   = log_on_step,
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
                    on_epoch   = log_on_epoch,
                    on_step    = log_on_step,
                    sync_dist  = True
                )
        
        if not is_training or step_data is None:
            if computed := self._compute_ready_metrics(
                self._get_metrics(is_training, "evaluation")
            ):
                prefixed = {f"{phase}/{k}": v for k, v in computed.items()}
                module.log_dict(
                    dictionary = prefixed,
                    on_epoch   = log_on_epoch,
                    on_step    = log_on_step,
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

        if all(k in batch for k in ["position", "velocity"]):
            metrics["hamiltonian"].update(batch)

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
            for name in [
                "connectivity", "dynamic_balance", "info_propagation",
                "scale_free", "state", "susceptibility", "topo_fidelity"
            ]:
                if name in metrics:
                    metrics[name].update(batch)
    
