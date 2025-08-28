"""
Unified metrics collection for imitation learning training and evaluation.

This module provides a centralized MetricsCollector that manages all metrics
for the training pipeline, including imitation learning losses, core evaluation
metrics, and runtime performance tracking. The collector integrates seamlessly
with PyTorch Lightning's logging system and Weights & Biases.
"""
from __future__           import annotations
from collections          import Counter
from itertools            import pairwise
from torch_geometric.data import Batch
from torchmetrics         import MeanAbsoluteError, MeanSquaredError
from torchmetrics         import Metric, MetricCollection, R2Score
from typing               import TYPE_CHECKING

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

    Provides common functionality for all averaging metrics including:
    - State management (sum/count) for distributed training
    - PyG batch operations (reshaping, batch size extraction)
    - Common configuration storage (agent_count, mmm, metrics, etc.)
    - Helper methods for tensor operations
    """
    agent_count : int | None
    count       : Tensor
    sum         : Tensor

    def __init__(
        self,
        agent_count : int | None = None,
        gravity     : float | None = None,
        metrics     : MetricsModel | None = None,
        mmm         : MurmurationModel | None = None,
        safety      : SafetyModel | None = None,
        **kwargs
    ):
        """
        Initialize state variables and common configuration.

        Creates sum/count states for averaging and stores common configs
        that child metrics need. Child classes can access any of these
        via self attributes.
        
        Args:
            agent_count : Number of agents in flock for tensor reshaping
            gravity     : Gravitational acceleration for physics calculations
            metrics     : Metrics configuration model
            mmm         : Murmuration dynamics configuration  
            safety      : Safety configuration with thresholds
            **kwargs    : Additional parameters for child classes
        """
        super().__init__()
        self.agent_count = agent_count
        self.gravity     = gravity
        self.metrics     = metrics
        self.mmm         = mmm
        self.safety      = safety
        
        # Store any additional kwargs for child classes
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self.add_state("count", th.tensor(0),   "sum")
        self.add_state("sum",   th.tensor(0.0), "sum")
    
    def _get_batch_info(self, batch: Batch) -> tuple[int, int]:
        """
        Extract batch size and agent count from PyG batch.
        
        Args:
            batch : PyG Batch object
            
        Returns:
            Tuple of (batch_size, agent_count)
        """
        batch_size = getattr(batch, 'num_graphs', 1)
        agent_count = self.agent_count or (
            batch["position"].shape[0] // batch_size 
            if "position" in batch else None
        )
        return batch_size, agent_count
    
    def _reshape_features(
        self, 
        batch      : Batch,
        *features  : str
    ) -> tuple[Tensor, ...]:
        """
        Reshape flattened PyG features to [B, N, F] format.
        
        Args:
            batch    : PyG Batch containing the features
            features : Names of features to reshape
            
        Returns:
            Tuple of reshaped tensors matching the order of feature names
        """
        batch_size, agent_count = self._get_batch_info(batch)
        
        return tuple(
            batch[feat].view(batch_size, agent_count, -1)
            for feat in features
            if feat in batch
        )
    
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
    
    Expected values:
        - Disconnected     : λ₂ = 0
        - Weakly connected : λ₂ ∈ (0, 0.1]
        - Well connected   : λ₂ ∈ (0.1, 0.5]
        - Strongly connected: λ₂ > 0.5
    """
    
    def __init__(self, **kwargs):
        """
        Initialize with flock configuration.
        
        All configuration passed from MetricsCollector via kwargs.
        """
        super().__init__(**kwargs)

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

    def update(self, batch: Batch):
        """
        Update metric with graph connectivity measurement.

        Computes the Fiedler value (second-smallest eigenvalue) of the
        graph Laplacian to quantify algebraic connectivity.

        Args:
            batch : PyG Batch containing edge_index [2, E] in COO format
        """
        if not (edge_index := batch.get("edge_index")) or edge_index.numel() == 0:
            self.sum   += 0.0
            self.count += 1
            return

        if edge_index.dim() == 3:
            batch_size = edge_index.shape[0]
            fiedler_values = th.zeros(batch_size, device=edge_index.device)
            
            for i in range(batch_size):
                if edge_index[i].numel() > 0:
                    laplacian = self._compute_graph_laplacian(
                        edge_index[i], self.agent_count
                    )
                    fiedler_values[i] = self._compute_fiedler_power_iteration(laplacian)
            
            self.sum   += fiedler_values.sum()
            self.count += batch_size
            return

        laplacian = self._compute_graph_laplacian(edge_index, self.agent_count)
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
        super().__init__(gravity=gravity, metrics=metrics)
        self.register_buffer("gravity_tensor", th.tensor(gravity))
        self.power_exponent = metrics.power_exponent

    def update(self, u_safe: Tensor):
        """
        Computes instantaneous power from thrust vector magnitude.

        Args:
            u_safe : Safety-filtered control actions [N, 3] (m/s²)
        """
        gravity_vector         = th.zeros_like(u_safe)
        gravity_vector[..., 2] = -self.gravity_tensor
        thrust_magnitude       = (u_safe - gravity_vector).norm(dim=-1)

        power_sum = thrust_magnitude.pow(self.power_exponent).sum()
        self.sum   += power_sum
        self.count += u_safe.shape[0]


class HamiltonianEnergyMetric(AveragingMetric):
    """
    Track Hamiltonian energy E = -Σ_{⟨ij⟩} J_{ij} 𝐬ᵢ·𝐬ⱼ per timestep.

    Computes the interaction energy of the flock using physics-inspired
    spin glass formulation where agents are spins with pairwise coupling:
    
        H = -Σᵢⱼ J_{ij}^{alert} (v̂ᵢ · v̂ⱼ)
    
    where J_{ij}^{alert} = κᵢ J₀ exp(-dᵢⱼ/λ) with:
        - κᵢ = 1.0 for relaxed birds, α for alert birds
        - dᵢⱼ is topological distance from k-NN graph
        - λ is the coupling decay length
    
    Energy minimization drives alignment (E < 0) while thermal fluctuations
    (alert states) increase disorder, creating rich phase transitions.
    """

    def __init__(
        self,
        agent_count : int,
        mmm         : MurmurationModel
    ):
        """
        Initialize with murmuration model parameters.

        Args:
            agent_count : Number of agents for tensor reshaping
            mmm         : Murmuration model with coupling parameters
        """
        super().__init__(agent_count=agent_count, mmm=mmm)
        self.j_base                = mmm.j_base
        self.coupling_decay        = mmm.coupling_decay
        self.alert_coupling_factor = mmm.alert_coupling_factor

    def update(self, batch: Batch):
        """
        Compute Hamiltonian energy with vectorized operations.
        
        Efficiently computes spin-spin interactions using batched matrix
        operations optimized for MPS/GPU execution.

        Args:
            batch: PyG Batch with velocity, position, optional alert_states
        """
        batch_size, _ = self._get_batch_info(batch)
        velocities, = self._reshape_features(batch, "velocity")
        spins = th.nn.functional.normalize(velocities, dim=-1)
        
        if "topo_distances" in batch and "edge_source" in batch:
            coupling = th.zeros(
                batch_size, self.agent_count, self.agent_count, 
                device=spins.device
            )
            
            if (n_edges := batch.get("edge_source", th.empty(0)).shape[-1]) > 0:
                batch_idx = (
                    th.arange(batch_size, device=spins.device)
                    .unsqueeze(1).expand(-1, n_edges)
                )
                
                alert_factor = (
                    self.alert_coupling_factor 
                    if "alert_states" in batch and (
                        batch["alert_states"][batch_idx, batch["edge_source"]] > 0.5
                    ).any() else 1.0
                )
                
                j_edges = self.j_base * alert_factor * th.exp(
                    -batch["topo_distances"][
                        batch_idx, batch["edge_source"], batch["edge_target"]
                    ] / self.coupling_decay
                )
                
                coupling[batch_idx, batch["edge_source"], batch["edge_target"]] = j_edges
                coupling[batch_idx, batch["edge_target"], batch["edge_source"]] = j_edges
        else:
            positions, = self._reshape_features(batch, "position")
            distances = th.cdist(positions, positions)
            coupling  = self.j_base * th.exp(-distances / self.coupling_decay)
            coupling.diagonal(dim1=-2, dim2=-1).fill_(0)

        energies = -(coupling * th.bmm(spins, spins.mT)).sum(dim=(1, 2)) / 2
        
        self.sum   += energies.sum()
        self.count += batch_size


class NeighborStabilityMetric(AveragingMetric):
    """
    Quantify topological stability of the communication graph.
    
    Measures the Jaccard distance between consecutive graph snapshots to
    track neighborhood relationship changes over time. The metric computes:
    
        Δ_topo = 1 - J(E_t, E_{t-1}) = |E_t ∆ E_{t-1}| / |E_t ∪ E_{t-1}|
    
    where E_t is the edge set at time t and ∆ denotes symmetric difference.
    
    Lower values (Δ_topo → 0) indicate stable flocking structure with
    persistent neighborhoods, while higher values (Δ_topo → 1) suggest
    rapid reconfiguration typical of threat evasion or murmuration waves.
    
    Expected ranges:
        - Cruising flight  : Δ_topo ∈ [0.0, 0.1]
        - Turning maneuver : Δ_topo ∈ [0.1, 0.3]
        - Threat response  : Δ_topo ∈ [0.3, 0.6]
        - Murmuration      : Δ_topo ∈ [0.4, 0.8]
    """
    
    def __init__(self, agent_count : int):
        """
        Initialize with flock configuration.
        
        Args:
            agent_count: Number of agents for edge normalization
        """
        super().__init__(agent_count=agent_count)
        self.last_edges = None
    
    def update(self, batch: Batch):
        """
        Update metric with topological change measurement.
        
        Efficiently computes edge set differences using vectorized operations
        for optimal MPS/GPU performance.
        
        Args:
            batch: PyG Batch containing edge_index [2, E] in COO format
        """
        if not (edges := batch.get("edge_index")) or edges.numel() == 0:
            self.last_edges = th.empty(0, 2, dtype=th.long)
            return
        
        current_edges = th.unique(edges.T, dim=0)
        
        if self.last_edges is not None and self.last_edges.numel() > 0:
            unique_edges, counts = th.unique(
                th.cat([current_edges, self.last_edges]), 
                dim=0, 
                return_counts=True
            )
            
            if union_size := unique_edges.shape[0]:
                jaccard_distance = 1.0 - (counts == 2).sum().item() / union_size
                self.sum   += jaccard_distance
                self.count += 1
        
        self.last_edges = current_edges


class OrientationCoherenceMetric(AveragingMetric):
    """
    Quantify directional alignment coherence via order parameter Φ.
    
    Computes the polarization order parameter measuring collective alignment
    of velocity vectors in the horizontal plane:
    
        Φ = |⟨ŝᵢ⟩| = |Σᵢ v̂ᵢ| / N
    
    where v̂ᵢ = vᵢ/|vᵢ| are normalized 2D velocity projections. This metric
    captures the phase transition between disordered (Φ ≈ 0) and ordered
    (Φ ≈ 1) collective motion states.
    
    For pairwise coherence, we compute:
    
        C = ⟨v̂ᵢ · v̂ⱼ⟩_{i≠j} = (Σᵢⱼ cos θᵢⱼ) / (N(N-1))
    
    Expected values:
        - Random flight     : Φ ∈ [0.0, 0.2], C ≈ 0
        - Loose aggregation : Φ ∈ [0.2, 0.5], C ∈ [0.1, 0.3]
        - Coordinated turn  : Φ ∈ [0.5, 0.8], C ∈ [0.3, 0.6]
        - Aligned cruise    : Φ ∈ [0.8, 1.0], C ∈ [0.6, 1.0]
    """
    
    def __init__(self, agent_count : int):
        """
        Initialize with flock configuration.
        
        Args:
            agent_count: Number of agents for tensor reshaping
        """
        super().__init__(agent_count=agent_count)
    
    def update(self, batch: Batch):
        """
        Update metric with polarization measurement.
        
        Uses batched matrix multiplication for efficient computation on
        MPS/GPU, avoiding explicit loops over agent pairs.
        
        Args:
            batch: PyG Batch containing velocity [B*N, 3] flattened
        """
        if not (velocity := batch.get("velocity")):
            return
        
        batch_size, _ = self._get_batch_info(batch)
        velocities, = self._reshape_features(batch, "velocity")
        headings = th.nn.functional.normalize(velocities[:, :, :2], dim=-1)
        
        alignment = th.bmm(headings, headings.mT)
        coherence = (
            alignment.sum(dim=(1, 2)) - batch_size * self.agent_count
        ) / (self.agent_count * (self.agent_count - 1))
        
        self.sum   += coherence.sum()
        self.count += batch_size


class OrientationWaveMetric(AveragingMetric):
    """
    Detect traveling waves in the orientation field ∇θ(𝐫, t).
    
    Identifies density waves characteristic of murmurations by computing
    spatial gradients of heading angles. The metric measures:
    
        W = ⟨|∇θᵢ|⟩ = ⟨|dθ/dr|⟩_local
    
    where θᵢ = atan2(vᵧ, vₓ) is the heading angle and gradients are
    computed over local neighborhoods within radius R_wave.
    
    Traveling waves manifest as coherent rotation patterns propagating
    through the flock with characteristic wavelength λ ≈ 7-10 body lengths
    and phase velocity c ≈ 0.3-0.5 v_flock (Attanasi et al. 2014).
    
    Expected values:
        - Straight flight  : W ∈ [0.00, 0.05] rad/m
        - Collective turn  : W ∈ [0.05, 0.15] rad/m
        - Density wave     : W ∈ [0.15, 0.40] rad/m
        - Full murmuration : W ∈ [0.30, 0.60] rad/m
    """
    
    def __init__(
        self,
        agent_count : int,
        mmm         : MurmurationModel
    ):
        """
        Initialize with murmuration configuration.
        
        Args:
            agent_count : Number of agents in flock
            mmm         : Murmuration model with wave detection radius
        """
        super().__init__(agent_count=agent_count, mmm=mmm)
        self.wave_radius = getattr(mmm, 'wave_radius', 10.0)
    
    def update(self, batch: Batch):
        """
        Update metric with wave amplitude measurement.
        
        Uses vectorized distance computations and masked operations for
        efficient gradient calculation on MPS/GPU.
        
        Args:
            batch: PyG Batch with position and velocity [B*N, 3] flattened
        """
        if not all(batch.get(k) is not None for k in ["position", "velocity"]):
            return
        
        batch_size, _ = self._get_batch_info(batch)
        positions, velocities = self._reshape_features(batch, "position", "velocity")
        
        headings  = th.atan2(velocities[..., 1], velocities[..., 0])
        distances = th.cdist(positions, positions)
        
        mask = (distances > 0) & (distances < self.wave_radius)
        
        heading_diffs = (
            lambda h: th.remainder(h + th.pi, 2 * th.pi) - th.pi
        )(headings.unsqueeze(-1) - headings.unsqueeze(-2))
        
        gradients = (
            heading_diffs.abs() / distances.clamp_min(1e-6)
        ).masked_fill(~mask, 0)
        
        self.sum   += gradients.sum(dim=(1, 2)).mean(dim=0)
        self.count += batch_size


class PerturbationResponseMetric(AveragingMetric):
    """
    Quantify collective response to thermal perturbations χ_thermal.
    
    Measures information propagation efficiency by tracking velocity response
    amplification from threatened to safe agents:
    
        χ_thermal = ⟨|Δ𝐯_safe|⟩ / ⟨|Δ𝐯_threat|⟩
    
    where Δ𝐯 = 𝐯(t) - 𝐯(t-Δt) represents velocity changes between timesteps.
    
    This susceptibility metric quantifies the flock's ability to amplify and
    propagate threat information through the interaction network, critical
    for collective evasion maneuvers.
    
    Expected response ratios:
        - No propagation       : χ ∈ [0.0, 0.1]
        - Weak coupling        : χ ∈ [0.1, 0.3]
        - Critical regime      : χ ∈ [0.3, 0.7]
        - Strong amplification : χ ∈ [0.7, 1.5]
    """
    
    def __init__(
        self, 
        agent_count : int,
        safety      : SafetyModel
    ):
        """
        Initialize with threat detection parameters.
        
        Args:
            agent_count : Number of agents for normalization
            safety      : Safety model with temperature thresholds
        """
        super().__init__(agent_count=agent_count, safety=safety)
        self.threat_threshold = safety.max_temperature
        self.last_velocity   = None
    
    def update(self, batch: Batch):
        """
        Update metric with threat response measurement.
        
        Efficiently computes response ratios using masked tensor operations
        for optimal GPU performance.
        
        Args:
            batch: PyG Batch with velocity [B*N, 3] and temperature [B*N, 1]
        """
        if not (
            (velocity    := batch.get("velocity")) and 
            (temperature := batch.get("temperature"))
        ):
            return
        
        if (
            self.last_velocity is not None and 
            velocity.shape == self.last_velocity.shape and
            (threat_mask := temperature.squeeze(-1) > self.threat_threshold).any() and
            (~threat_mask).any()
        ):
            vel_changes = (velocity - self.last_velocity).norm(dim=-1)
            
            if (threat_response := vel_changes[threat_mask].mean()) > 1e-6:
                self.sum   += vel_changes[~threat_mask].mean() / threat_response
                self.count += 1
        
        self.last_velocity = velocity.detach().clone()


class PowerComponentsMetric(Metric):
    """
    Decompose power P = Σᵢ ||𝐮ᵢ||^k into physical flight components.
    
    Provides actionable energy breakdown following quadrotor power model
    (Hoffmann et al. 2011) where power scales with thrust magnitude:
    
        P = P_hover + P_forward + P_lateral
    
    Components computed via orthogonal decomposition:
        - P_hover   = ||u_z + g||^k    : Anti-gravity thrust
        - P_forward = ||𝐮 · v̂||^k      : Along-velocity thrust
        - P_lateral = ||𝐮 - (𝐮·v̂)v̂||^k : Perpendicular thrust
    
    where k ≈ 1.5 for quadrotors in hover-dominant regimes.
    
    Expected power distribution:
        - Hovering    : P_h ≈ 70%, P_f ≈ 10%, P_l ≈ 20%
        - Cruising    : P_h ≈ 40%, P_f ≈ 50%, P_l ≈ 10%
        - Maneuvering : P_h ≈ 50%, P_f ≈ 20%, P_l ≈ 30%
        - Murmuration : P_h ≈ 45%, P_f ≈ 25%, P_l ≈ 30%
    """
    
    def __init__(
        self,
        agent_count : int,
        gravity     : float,
        metrics     : MetricsModel
    ):
        """
        Initialize with physics and power model parameters.
        
        Args:
            agent_count : Number of agents for normalization
            gravity     : Gravitational acceleration [m/s²]
            metrics     : Metrics config with power exponent k
        """
        super().__init__()
        self.agent_count    = agent_count
        self.gravity        = gravity
        self.power_exponent = metrics.power_exponent
        
        self.add_state("count",         th.tensor(0),   "sum")
        self.add_state("power_forward", th.tensor(0.0), "sum")
        self.add_state("power_hover",   th.tensor(0.0), "sum")
        self.add_state("power_lateral", th.tensor(0.0), "sum")
    
    def compute(self) -> dict[str, Tensor]:
        """
        Compute normalized power component averages.
        
        Returns:
            Dictionary with hover, forward, and lateral power fractions
        """
        if self.count == 0:
            zero = th.tensor(0.0)
            return {
                "power_forward" : zero,
                "power_hover"   : zero,
                "power_lateral" : zero,
            }
        
        count = self.count.float()
        return {
            "power_forward" : self.power_forward / count,
            "power_hover"   : self.power_hover   / count,
            "power_lateral" : self.power_lateral / count,
        }
    
    def update(self, batch: Batch):
        """
        Update power component measurements with vectorized computation.
        
        Efficiently decomposes control forces using batched operations
        optimized for MPS/GPU execution.
        
        Args:
            batch: PyG Batch with action [B*N, 3] and velocity [B*N, 3]
        """
        if not (
            (u_control := batch.get("u_safe") or batch.get("action")) and
            (velocity  := batch.get("velocity"))
        ):
            return
        
        self.power_hover += (
            u_control[:, 2] + self.gravity
        ).abs().pow(self.power_exponent).sum()
        
        if (mask := velocity.norm(dim=-1) > 1e-3).any():
            v_hat     = th.nn.functional.normalize(velocity[mask], dim=-1)
            u_masked  = u_control[mask]
            forward   = (u_masked * v_hat).sum(dim=-1).clamp_min(0)
            
            self.power_forward += forward.pow(self.power_exponent).sum()
            self.power_lateral += (
                u_masked - forward.unsqueeze(-1) * v_hat
            ).norm(dim=-1).pow(self.power_exponent).sum()
        
        self.count += u_control.shape[0]


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
    
    def __init__(
        self,
        agent_count : int,
        metrics     : MetricsModel
    ):
        """
        Initialize with target correlation exponent.
        
        Args:
            agent_count : Number of agents for tensor reshaping
            metrics     : Metrics model with expected exponent γ
        """
        super().__init__(agent_count=agent_count, metrics=metrics)
        self.target_exponent = self.metrics.correlation_exponent
    
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
        batch_size, _ = self._get_batch_info(batch)
        positions, velocities = self._reshape_features(batch, "position", "velocity")
        
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
    
    def __init__(
        self,
        agent_count : int,
        metrics     : MetricsModel
    ):
        """
        Initialize with susceptibility configuration.
        
        Args:
            agent_count : Number of agents for tensor reshaping
            metrics     : Metrics configuration with susceptibility range
        """
        super().__init__(agent_count=agent_count, metrics=metrics)
        self.target_min = self.metrics.susceptibility_min
        self.target_max = self.metrics.susceptibility_max
    
    def update(self, batch: Batch):
        """
        Compute susceptibility from velocity fluctuations.
        
        Args:
            batch: PyG Batch containing velocity tensor [B*N, 3] flattened
        """
        if self.agent_count < 2:
            return
        
        batch_size, _ = self._get_batch_info(batch)
        velocity, = self._reshape_features(batch, "velocity")
        spins      = th.nn.functional.normalize(velocity, dim=-1)
        mean_spin  = spins.mean(dim=-2, keepdim=True)
        
        polarizations    = (spins * mean_spin).sum(dim=-1)
        susceptibilities = self.agent_count * polarizations.var(dim=-1)
        
        self.sum   += susceptibilities.sum()
        self.count += susceptibilities.numel()


class MetricsCollector(th.nn.Module):
    """
    Centralized metric collection and management for training and evaluation.

    This class manages all TorchMetrics instances for the training pipeline,
    working with PyG's flattened batch format where graphs are concatenated
    as [B*N, F] tensors instead of hierarchical [B, N, F] format.

    Integrates with PyTorch Lightning's logging system and Weights & Biases.
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
        
        if readiness[True] == 0:
            return None
        
        try:
            computed = metrics.compute()
            return {
                k: v for k, v in computed.items()
                if v is not None and not th.isnan(v).any()
            }
        except Exception:
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
        step-wise during training. All metrics work with PyG's flattened
        batch format where agent features are concatenated as [B*N, F].
        
        Metrics include:
        - Regression        : MSE, RMSE, MAE, R² for velocity prediction
        - Graph topology    : Fiedler value for cohesion, edge stability
        - Emergent dynamics : Scale-free correlations, susceptibility
        - Energy            : Hamiltonian energy, power component breakdown
        - Murmuration       : Orientation coherence, density wave detection
        - Threat response   : Perturbation propagation metrics
        - Physical states   : Velocity, temperature, acceleration averages
        """
        cfg = {
            'agent_count' : self.agent_count,
            'gravity'     : self.gravity,
            'metrics'     : self.metrics,
            'mmm'         : self.mmm,
            'safety'      : self.safety,
        }
        
        self.train_metrics = MetricCollection({
            "cohesion_fiedler_value" : CohesionMetric(**cfg),
            "hamiltonian_energy"     : HamiltonianEnergyMetric(**cfg),
            "mae"                    : MeanAbsoluteError(),
            "mse"                    : MeanSquaredError(),
            "neighbor_stability"     : NeighborStabilityMetric(**cfg),
            "orientation_coherence"  : OrientationCoherenceMetric(**cfg),
            "orientation_wave"       : OrientationWaveMetric(**cfg),
            "perturbation_response"  : PerturbationResponseMetric(**cfg),
            "power_components"       : PowerComponentsMetric(**cfg),
            "r2"                     : R2Score(multioutput='uniform_average'),
            "rmse"                   : MeanSquaredError(squared=False),
            "scale_free"             : ScaleFreeCorrelationMetric(**cfg),
            "state"                  : StateMetrics(),
            "susceptibility"         : SusceptibilityMetric(**cfg),
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
        if batch is None and predictions is None:
            return
        
        metrics = self.train_metrics if is_training else self.val_metrics
        
        if predictions is not None and targets is not None:
            for name in ["mae", "mse", "r2", "rmse"]:
                if name in metrics:
                    try:
                        metrics[name].update(predictions, targets)
                    except Exception:
                        pass
        
        if batch is not None:
            if all(hasattr(batch, k) for k in ["position", "velocity"]):
                try:
                    metrics["hamiltonian_energy"].update(batch)
                except Exception:
                    pass
            
            if hasattr(batch, "edge_index"):
                try:
                    metrics["cohesion_fiedler_value"].update(batch)
                except Exception:
                    pass
            
            for name in [
                "neighbor_stability",
                "orientation_coherence", 
                "orientation_wave",
                "perturbation_response",
                "power_components",
                "scale_free",
                "state",
                "susceptibility"
            ]:
                if name in metrics:
                    try:
                        metrics[name].update(batch)
                    except Exception:
                        pass
    
