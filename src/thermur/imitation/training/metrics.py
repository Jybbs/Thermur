"""
Unified metrics for imitation learning training and evaluation.

This module provides TorchMetrics-based metrics that work with PyTorch 
Geometric's batch format where features are flattened as [B*N, F] tensors 
(B=batch_size, N=agents, F=features). The BaseMetric base class extends
MeanMetric to provide automatic averaging and PyG batch utilities.

All metrics integrate seamlessly with PyTorch Lightning's logging system
and can be used directly in LightningModules without a separate collector.
"""
from __future__           import annotations
from collections          import defaultdict
from itertools            import pairwise
from torch_geometric.data import Batch
from torchmetrics         import MeanAbsoluteError, MeanMetric, MeanSquaredError
from torchmetrics         import MetricCollection, R2Score
from typing               import TYPE_CHECKING

if TYPE_CHECKING:
    from config.imitation.controller  import MurmurationModel, SafetyModel
    from config.imitation.environment import PhysicsModel
    from config.imitation.training    import MetricsModel
    from torch                        import Tensor

import torch as th


class BaseMetric(MeanMetric):
    """
    Base class extending MeanMetric with PyG batch support.
    
    Provides automatic averaging from MeanMetric and PyG batch reshaping helpers.
    Metrics needing state history should use add_state() themselves.
    """

    def __init__(self, agent_count: int, **kwargs):
        """
        Initialize the base metric.
        
        Args:
            agent_count : Number of agents for reshaping
            **kwargs    : Additional config passed to child classes
        """
        super().__init__(nan_strategy='ignore')
        self.agent_count = agent_count
        
        for key, value in sorted(kwargs.items()):
            setattr(self, key, value)
    
    def _get_batch_info(self, batch: Batch) -> tuple[int, int | None]:
        """
        Extract batch size and agent count from PyG batch.
        
        Args:
            batch: PyG Batch object
            
        Returns:
            Tuple of (batch_size, agent_count)
        """
        batch_size  = getattr(batch, 'num_graphs', 1)
        agent_count = self.agent_count or (
            batch["position"].shape[0] // batch_size 
            if "position" in batch else None
        )
        return batch_size, agent_count
    
    def _reshape_features(
        self, 
        batch     : Batch,
        *features : str
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
            if  feat in batch
        )


class FiedlerValueMetric(BaseMetric):
    """
    Measure graph connectivity via the Fiedler value λ₂.

    The algebraic connectivity quantifies how well-connected the flock's
    communication graph is. Higher values indicate stronger cohesion, with
    λ₂ = 0 for disconnected graphs and λ₂ > 0 for connected components.

    The metric computes the second-smallest eigenvalue of the graph Laplacian:

        L = D - A

    where D is the degree matrix and A is the adjacency matrix.
    
    Expected values:
        - Disconnected       : λ₂ = 0
        - Weakly connected   : λ₂ ∈ (0, 0.1]
        - Well connected     : λ₂ ∈ (0.1, 0.5]
        - Strongly connected : λ₂ > 0.5
    """

    def _compute_fiedler_power_iteration(
        self,
        laplacian : Tensor
    ) -> Tensor:
        """
        Compute Fiedler value using power iteration method.
        
        MPS-compatible alternative to eigvalsh that avoids CPU fallback.
        Uses inverse power iteration with shift to find second smallest eigenvalue.
        
        Args:
            laplacian: Graph Laplacian matrix [n, n]
            
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
        
        shifted_laplacian = laplacian + self.fiedler_shift * th.eye(n, device=device)
        
        for _ in range(self.power_iterations):
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
        if not (edge_index := getattr(batch, "edge_index", None)) or edge_index.numel() == 0:
            super().update(0.0)
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
            
            for value in fiedler_values:
                super().update(value)
            return

        laplacian = self._compute_graph_laplacian(edge_index, self.agent_count)
        fiedler_value = self._compute_fiedler_power_iteration(laplacian)
        
        super().update(fiedler_value)


class HamiltonianEnergyMetric(BaseMetric):
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

    def update(self, batch: Batch):
        """
        Compute Hamiltonian energy with vectorized operations.
        
        Efficiently computes spin-spin interactions using batched matrix
        operations optimized for MPS/GPU execution.

        Args:
            batch: PyG Batch with velocity, position, optional alert_states
        """
        batch_size, _ = self._get_batch_info(batch)
        velocities,   = self._reshape_features(batch, "velocity")
        spins         = th.nn.functional.normalize(velocities, dim=-1)
        
        if "topo_distances" in batch and "edge_source" in batch:
            coupling = th.zeros(
                batch_size, self.agent_count, self.agent_count, 
                device = spins.device
            )
            
            if (n_edges := getattr(batch, "edge_source", th.empty(0)).shape[-1]) > 0:
                idx = (
                    th.arange(batch_size, device=spins.device)
                    .unsqueeze(1).expand(-1, n_edges)
                )
                
                alert_factor = (
                    self.alert_coupling_factor 
                    if "alert_states" in batch and (
                        batch["alert_states"][idx, batch["edge_source"]] > 0.5
                    ).any() else 1.0
                )
                
                j_edges = self.j_base * alert_factor * th.exp(
                    -batch["topo_distances"][
                        idx, batch["edge_source"], batch["edge_target"]
                    ] / self.coupling_decay
                )
                
                coupling[idx, batch["edge_source"], batch["edge_target"]] = j_edges
                coupling[idx, batch["edge_target"], batch["edge_source"]] = j_edges
        else:
            positions, = self._reshape_features(batch, "position")
            distances = th.cdist(positions, positions)
            coupling  = self.j_base * th.exp(-distances / self.coupling_decay)
            coupling.diagonal(dim1=-2, dim2=-1).fill_(0)

        energies = -(coupling * th.bmm(spins, spins.mT)).sum(dim=(1, 2)) / 2
        
        for energy in energies:
            super().update(energy)


class NeighborStabilityMetric(BaseMetric):
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
    
    def __init__(self, **kwargs):
        """
        Initialize neighbor stability metric.
        
        Stores last edges for computing Jaccard distance.
        """
        super().__init__(**kwargs)
        self.add_state("last_edges", None, None, persistent=False)
    
    def update(self, batch: Batch):
        """
        Update metric with topological change measurement.
        
        Efficiently computes edge set differences using vectorized operations
        for optimal MPS/GPU performance.
        
        Args:
            batch: PyG Batch containing edge_index [2, E] in COO format
        """
        if not (edges := getattr(batch, "edge_index", None)) or edges.numel() == 0:
            self.last_edges = th.empty(0, 2, dtype=th.long)
            return
        
        current_edges = th.unique(edges.T, dim=0)
        
        if self.last_edges is not None and self.last_edges.numel() > 0:
            unique_edges, counts = th.unique(
                th.cat([current_edges, self.last_edges]), 
                dim           = 0, 
                return_counts = True
            )
            
            if union_size := unique_edges.shape[0]:
                jaccard_distance = 1.0 - (counts == 2).sum().item() / union_size
                super().update(jaccard_distance)
        
        self.last_edges = current_edges


class OrientationCoherenceMetric(BaseMetric):
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

    def update(self, batch: Batch):
        """
        Update metric with polarization measurement.
        
        Uses batched matrix multiplication for efficient computation on
        MPS/GPU, avoiding explicit loops over agent pairs.
        
        Args:
            batch: PyG Batch containing velocity [B*N, 3] flattened
        """
        if not (velocity := getattr(batch, "velocity", None)):
            return
        
        batch_size, _ = self._get_batch_info(batch)
        velocities,   = self._reshape_features(batch, "velocity")
        headings      = th.nn.functional.normalize(velocities[:, :, :2], dim=-1)
        
        alignment = th.bmm(headings, headings.mT)
        coherence = (
            alignment.sum(dim=(1, 2)) - batch_size * self.agent_count
        ) / (self.agent_count * (self.agent_count - 1))
        
        for c in coherence:
            super().update(c)


class OrientationWaveMetric(BaseMetric):
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

    def update(self, batch: Batch):
        """
        Update metric with wave amplitude measurement.
        
        Uses vectorized distance computations and masked operations for
        efficient gradient calculation on MPS/GPU.
        
        Args:
            batch: PyG Batch with position and velocity [B*N, 3] flattened
        """
        if not all(getattr(batch, k, None) is not None for k in ["position", "velocity"]):
            return
        
        batch_size, _ = self._get_batch_info(batch)
        positions, velocities = self._reshape_features(batch, "position", "velocity")
        
        headings  = th.atan2(velocities[..., 1], velocities[..., 0])
        distances = th.cdist(positions, positions)
        mask      = (distances > 0) & (distances < self.wave_radius)
        
        heading_diffs = (
            lambda h: th.remainder(h + th.pi, 2 * th.pi) - th.pi
        )(headings.unsqueeze(-1) - headings.unsqueeze(-2))
        
        gradients = (
            heading_diffs.abs() / distances.clamp_min(self.epsilon)
        ).masked_fill(~mask, 0)
        
        super().update(gradients.sum(dim=(1, 2)).mean(dim=0))


class PerturbationResponseMetric(BaseMetric):
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

    def __init__(self, **kwargs):
        """
        Initialize perturbation response metric.
        
        Stores last velocity for computing response ratios.
        """
        super().__init__(**kwargs)
        self.add_state("last_velocity", None, None, persistent=False)

    def update(self, batch: Batch):
        """
        Update metric with threat response measurement.
        
        Efficiently computes response ratios using masked tensor operations
        for optimal GPU performance.
        
        Args:
            batch: PyG Batch with velocity [B*N, 3] and temperature [B*N, 1]
        """
        if not (
            (velocity    := getattr(batch, "velocity", None)) and 
            (temperature := getattr(batch, "temperature", None))
        ):
            return
        
        if (
            self.last_velocity is not None 
            and velocity.shape == self.last_velocity.shape 
            and (threat_mask := temperature.squeeze(-1) > self.max_temperature).any() 
            and (~threat_mask).any()
        ):
            vel_changes = (velocity - self.last_velocity).norm(dim=-1)
            
            if (threat_response := vel_changes[threat_mask].mean()) > self.epsilon:
                non_threat_response = vel_changes[~threat_mask].mean()
                super().update(non_threat_response / threat_response)
        
        self.last_velocity = velocity.detach().clone()


class PowerComponents(BaseMetric):
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

    def __init__(self, **kwargs):
        """
        Initialize power components metric.
        
        Stores all config via BaseMetric and creates component metrics.
        """
        super().__init__(**kwargs)
        
        self.forward = MeanMetric(nan_strategy='ignore')
        self.hover   = MeanMetric(nan_strategy='ignore')
        self.lateral = MeanMetric(nan_strategy='ignore')
    
    def compute(self) -> dict[str, Tensor]:
        """
        Compute power component fractions.
        
        Returns:
            Dictionary with hover, forward, and lateral power components
        """
        return {
            "power_forward" : self.forward.compute(),
            "power_hover"   : self.hover.compute(),
            "power_lateral" : self.lateral.compute(),
        }
    
    def reset(self):
        """
        Reset all component metrics.
        """
        self.forward.reset()
        self.hover.reset()
        self.lateral.reset()

    def update(self, batch: Batch):
        """
        Update power component measurements with vectorized computation.
        
        Efficiently decomposes control forces using batched operations
        optimized for MPS/GPU execution.
        
        Args:
            batch: PyG Batch with action [B*N, 3] and velocity [B*N, 3]
        """
        if not (
            (u_control := getattr(batch, "u_safe", None) or getattr(batch, "action", None)) and
            (velocity  := getattr(batch, "velocity", None))
        ):
            return
        
        hover_power = (u_control[:, 2] + self.gravity).abs().pow(self.power_exponent)
        self.hover.update(hover_power)
        
        mask = velocity.norm(dim=-1) > self.velocity_threshold
        if mask.any():
            v_hat     = th.nn.functional.normalize(velocity[mask], dim=-1)
            u_masked  = u_control[mask]
            forward   = (u_masked * v_hat).sum(dim=-1).clamp_min(0)
            
            forward_power = forward.pow(self.power_exponent)
            lateral_power = (
                u_masked - forward.unsqueeze(-1) * 
                v_hat
            ).norm(dim=-1).pow(self.power_exponent)
            
            full_forward       = th.zeros_like(hover_power)
            full_lateral       = th.zeros_like(hover_power)
            full_forward[mask] = forward_power
            full_lateral[mask] = lateral_power
            
            self.forward.update(full_forward)
            self.lateral.update(full_lateral)
        else:
            self.forward.update(th.tensor(0.0))
            self.lateral.update(th.tensor(0.0))


class ScaleFreeCorrelationMetric(BaseMetric):
    """
    Measure deviation from scale-free velocity correlations.
    
    Verifies that the flock exhibits power-law velocity correlations
    characteristic of critical systems. The correlation function C(r)
    should follow:

        C(r) ~ r^(-γ)

    where γ ≈ 1/3 for natural murmurations (Cavagna et al. 2010).
    """

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
            positions  : Agent positions  𝐱 ∈ ℝ^(n×3)
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
                
                super().update(abs(fitted_exponent - self.correlation_exponent))


class States(BaseMetric):
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
    def __init__(self, **kwargs):
        """
        Initialize state tracking for physical quantities.
        
        Composes multiple MeanMetrics for different state variables.
        """
        super().__init__(**kwargs)
        self.acceleration = MeanMetric(nan_strategy='ignore')
        self.temperature  = MeanMetric(nan_strategy='ignore')
        self.velocity     = MeanMetric(nan_strategy='ignore')
    
    def compute(self) -> dict[str, Tensor]:
        """
        Compute state averages.
        
        Returns:
            Dictionary with average acceleration, temperature, and velocity
        """
        return {
            "avg_acceleration" : self.acceleration.compute(),
            "avg_temperature"  : self.temperature.compute(),
            "avg_velocity"     : self.velocity.compute(),
        }
    
    def reset(self):
        """
        Reset all state metrics.
        """
        self.acceleration.reset()
        self.temperature.reset()
        self.velocity.reset()

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
        
        action = batch["action"]
        self.acceleration.update(action.norm(dim=-1))
        
        if "temperature" in batch and batch["temperature"].numel() > 0:
            self.temperature.update(batch["temperature"])
        
        if "velocity" in batch:
            self.velocity.update(batch["velocity"].norm(dim=-1))


class SusceptibilityMetric(BaseMetric):
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

    def update(self, batch: Batch):
        """
        Compute susceptibility from velocity fluctuations.
        
        Args:
            batch: PyG Batch containing velocity tensor [B*N, 3] flattened
        """
        if self.agent_count < 2:
            return
        
        batch_size, _ = self._get_batch_info(batch)
        velocity,     = self._reshape_features(batch, "velocity")
        spins         = th.nn.functional.normalize(velocity, dim=-1)
        mean_spin     = spins.mean(dim=-2, keepdim=True)
        
        polarizations    = (spins * mean_spin).sum(dim=-1)
        susceptibilities = self.agent_count * polarizations.var(dim=-1)
        
        super().update(susceptibilities)


class MetricsFactory:
    """
    Factory for creating training and validation metric collections.
    
    Centralizes metric instantiation with proper configuration from domain
    models. Creates both training and validation metrics with appropriate
    prefixes for PyTorch Lightning integration.
    """
    
    def __init__(
        self,
        agent_count : int,
        metrics     : MetricsModel,
        murmuration : MurmurationModel,
        physics     : PhysicsModel,
        safety      : SafetyModel
    ):
        """
        Initialize factory with configuration models.
        
        Args:
            agent_count : Number of agents in the flock
            environment : Environment configuration for physics parameters
            metrics     : Metrics configuration for thresholds and parameters
            murmuration : Murmuration dynamics configuration
            safety      : Safety configuration for temperature limits
        """
        self.cfg = {
            "agent_count"           : agent_count,
            "alert_coupling_factor" : murmuration.alert_coupling_factor,
            "correlation_exponent"  : metrics.correlation_exponent,
            "coupling_decay"        : murmuration.coupling_decay,
            "epsilon"               : metrics.epsilon,
            "fiedler_shift"         : metrics.fiedler_shift,
            "gravity"               : physics.gravity,
            "j_base"                : murmuration.j_base,
            "max_temperature"       : safety.max_temperature,
            "power_exponent"        : metrics.power_exponent,
            "power_iterations"      : metrics.power_iterations,
            "velocity_threshold"    : metrics.velocity_threshold,
            "wave_radius"           : metrics.wave_radius,
        }
    
    def create_training_metrics(self) -> MetricCollection:
        """
        Create training metric collection.
        
        Returns:
            MetricCollection with all configured metrics for training
        """
        make = lambda cls: cls(**self.cfg)
        
        return MetricCollection({
            "fiedler_value"          : make(FiedlerValueMetric),
            "hamiltonian_energy"     : make(HamiltonianEnergyMetric),
            "mae"                    : MeanAbsoluteError(),
            "mse"                    : MeanSquaredError(),
            "neighbor_stability"     : make(NeighborStabilityMetric),
            "orientation_coherence"  : make(OrientationCoherenceMetric),
            "orientation_wave"       : make(OrientationWaveMetric),
            "perturbation_response"  : make(PerturbationResponseMetric),
            "power_components"       : make(PowerComponents),
            "r2"                     : R2Score(num_outputs=3),
            "rmse"                   : MeanSquaredError(squared=False),
            "scale_free_correlation" : make(ScaleFreeCorrelationMetric),
            "states"                 : make(States),
            "susceptibility"         : make(SusceptibilityMetric),
        })
    
    def create_validation_metrics(self) -> MetricCollection:
        """
        Uses clone() to create a separate set of metrics for validation.
        
        Returns:
            MetricCollection with all configured metrics for validation
        """
        return self.create_training_metrics().clone()
