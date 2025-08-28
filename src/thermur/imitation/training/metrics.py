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
from torch_geometric.data import Batch
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

    def update(self, batch: Batch):
        """
        Compute Hamiltonian energy matching controller implementation.
        
        Implements the exact energy formulation from MurmurationController:
        E = -Σ_{<ij>} J_{ij}^{alert} 𝐬_i · 𝐬_j
        
        where J_{ij}^{alert} = κ_i × J_0 exp(-d_{ij}/λ) with:
        - κ_i = 1.0 for relaxed birds, alert_coupling_factor for alert birds  
        - d_{ij} is topological distance from k-NN graph
        - Only includes edges that exist in the controller's graph

        Args:
            batch: PyG Batch containing velocity, alert_states, edge indices, 
                   and topo_distances from controller
        """
        batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
        n_agents   = batch["velocity"].shape[0] // batch_size
        
        velocities = batch["velocity"].view(batch_size, n_agents, 3)
        spins      = th.nn.functional.normalize(velocities, dim=-1)
        
        if "topo_distances" not in batch or "edge_source" not in batch:
            positions = batch["position"].view(batch_size, n_agents, 3)
            coupling  = th.zeros(
                batch_size, n_agents, n_agents, device=spins.device
            )
            
            for b in range(batch_size):
                distances = th.cdist(positions[b], positions[b])
                coupling[b] = self.j_base * th.exp(
                    -distances / self.coupling_decay
                )
                coupling[b].diagonal().fill_(0)
        else:
            coupling = th.zeros(
                batch_size, n_agents, n_agents, device=spins.device
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
    
    def update(self, batch: Batch):
        """
        Update running sums with batch statistics.
        
        Extracts physical quantities from the batch and accumulates their
        magnitudes for computing running averages across all agents.
        Only tracks states when control actions are present.
        
        Args:
            batch: PyG Batch containing velocity, temperature, and optionally action
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

    def update(self, batch: Batch):
        """
        Update metric with scale-free correlation measurement.
        
        Computes velocity correlations C(r) as a function of distance r and fits
        power law C(r) ~ r^(-γ) to measure deviation from expected scaling. Uses
        adaptive logarithmic binning with n_bins ∈ [3, 10] based on flock size:
        
            n_bins = min(10, max(3, n_pairs // 10))
        
        Handles edge cases where d_min = d_max (no variation) or n_agents < 4
        (insufficient data for power law fitting).
        
        Args:
            batch: PyG Batch containing position and velocity [B*N, 3] flattened
        """
        batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
        n_agents   = batch["position"].shape[0] // batch_size
        
        positions  = batch["position"].view(batch_size, n_agents, 3)
        velocities = batch["velocity"].view(batch_size, n_agents, 3)
        
        for b in range(batch_size):
            corr_mat, distances = self._compute_velocity_correlations(
                positions[b], velocities[b]
            )
            
            triu_mask = th.triu(th.ones_like(distances), diagonal=1).bool()
            if not triu_mask.any():
                continue
            
            unique_distances = distances[triu_mask]
            max_dist         = unique_distances.max()
            min_dist         = unique_distances.min()
            
            if min_dist == max_dist:
                continue
            
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
    
    def update(self, batch: Batch):
        """
        Compute susceptibility from velocity fluctuations.
        
        Args:
            batch: PyG Batch containing velocity tensor [B*N, 3] flattened
        """
        batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
        n_agents   = batch["velocity"].shape[0] // batch_size
        
        if n_agents < 2:
            return
        
        velocity  = batch["velocity"].view(batch_size, n_agents, 3)
        spins     = th.nn.functional.normalize(velocity, dim=-1)
        mean_spin = spins.mean(dim=-2, keepdim=True)
        
        polarizations    = (spins * mean_spin).sum(dim=-1)
        susceptibilities = n_agents * polarizations.var(dim=-1)
        
        self.sum   += susceptibilities.sum()
        self.count += susceptibilities.numel()


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
        agent_count : int,
        bounds_max  : list[float],
        gravity     : float,
        metrics     : MetricsModel,
        mmm         : MurmurationModel,
        safety      : SafetyModel
    ):
        """
        Initialize metrics collector with configuration parameters.

        Creates all metric instances for tracking imitation learning,
        evaluation, and murmuration-specific measurements.

        Args:
            agent_count : Number of agents in the flock
            bounds_max  : Maximum workspace bounds [x_max, y_max, z_max]
            gravity     : Gravitational acceleration [m/s²]
            metrics     : Metrics configuration model
            mmm         : Murmuration dynamics configuration
            safety      : Safety configuration with temperature thresholds
        """
        super().__init__()
        self.agent_count = agent_count
        self.bounds_max  = bounds_max
        self.gravity     = gravity
        self.metrics     = metrics
        self.mmm         = mmm
        self.safety      = safety

        self._init_metrics()

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
            if isinstance(metric, (AveragingMetric, StateMetrics)):
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

    def _get_graph_view(
        self,
        feature    : Tensor,
        batch_size : int | None = None
    ) -> Tensor:
        """
        Convert flattened PyG tensor to [batch_size, n_agents, features].
        
        PyG provides flattened tensors where all graphs in a batch are
        concatenated. This method reshapes them back to the standard
        [batch, agents, features] format that many metrics expect, using
        zero-cost view operations.
        
        Args:
            feature    : Flattened tensor from PyG batch [B*N, F]
            batch_size : Number of graphs in batch, inferred if None
            
        Returns:
            Reshaped tensor [B, N, F] as a zero-cost view
        """
        return (
            feature if feature.dim() == 3
            else feature.view(
                batch_size or feature.shape[0] // self.agent_count,
                self.agent_count,
                -1
            )
        )

    def _get_metrics(self, is_training : bool) -> MetricCollection:
        """
        Get appropriate metrics collection for current phase.

        Returns the unified metrics collection for either training or
        validation phase.

        Args:
            is_training: Whether in training (True) or validation (False) phase

        Returns:
            MetricCollection for the specified phase
        """
        return self.train_metrics if is_training else self.val_metrics

    def _init_metrics(self):
        """
        Initialize unified metrics collection for training and evaluation.
        
        Creates a single collection containing all metrics that are logged
        step-wise during training. This eliminates the artificial distinction
        between "evaluation" and "imitation" metrics, allowing all metrics to
        be tracked continuously for better visibility.
        
        Metrics include:
        - Regression accuracy: MSE, RMSE, MAE, R²
        - Emergent behaviors: scale-free correlations, susceptibility
        - Energy dynamics: Hamiltonian energy, power consumption
        - Graph properties: cohesion (λ₂)
        - Physical states: velocity, temperature, acceleration averages
        """
        self.train_metrics = MetricCollection({
            "avg_power"      : EnergyConsumptionMetric(self.gravity, self.metrics),
            "hamiltonian"    : HamiltonianEnergyMetric(self.mmm),
            "λ₂"             : CohesionMetric(),
            "mae"            : MeanAbsoluteError(),
            "mse"            : MeanSquaredError(),
            "r2"             : R2Score(multioutput='uniform_average'),
            "rmse"           : MeanSquaredError(squared=False),
            "scale_free"     : ScaleFreeCorrelationMetric(self.metrics),
            "state"          : StateMetrics(),
            "susceptibility" : SusceptibilityMetric(self.metrics),
        })
        self.val_metrics = self.train_metrics.clone()
    
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
        
        metrics = self.train_metrics if is_training else self.val_metrics
        if computed := self._compute_ready_metrics(metrics):
            prefixed = {f"{phase}/{k}": v for k, v in computed.items()}
            module.log_dict(
                dictionary = prefixed,
                on_epoch   = log_on_epoch,
                on_step    = log_on_step,
                sync_dist  = True
            )

    def update_metrics(
        self,
        batch       : Batch | None,
        is_training : bool,
        predictions : Tensor | None = None,
        targets     : Tensor | None = None
    ):
        """
        Update all metrics from batch data and predictions.
        
        Unified method that updates both regression metrics (when predictions
        are provided) and state/dynamics metrics (when batch data is available).
        This eliminates the artificial distinction between metric types and
        ensures all metrics are updated consistently.
        
        Args:
            batch       : PyG Batch containing state, actions, and graph structure
            is_training : Whether in training (True) or validation (False) phase
            predictions : Model outputs for regression metrics [B*N, 3]
            targets     : Expert actions for regression metrics [B*N, 3]
        """
        metrics = self.train_metrics if is_training else self.val_metrics
        
        if predictions is not None and targets is not None:
            for name in ["mae", "mse", "r2", "rmse"]:
                if name in metrics:
                    metrics[name].update(predictions, targets)
        
        if batch is not None:
            if all(k in batch for k in ["position", "velocity"]):
                metrics["hamiltonian"].update(batch)
            
            if "edge_index" in batch:
                metrics["λ₂"].update(
                    edge_index = batch["edge_index"],
                    num_agents = self.agent_count
                )
            
            u_control = (
                batch.get("u_safe") or batch.get("action")
            )
            if u_control is not None:
                metrics["avg_power"].update(u_safe=u_control)
            
            for name in ["scale_free", "state", "susceptibility"]:
                if name in metrics:
                    metrics[name].update(batch)
    
