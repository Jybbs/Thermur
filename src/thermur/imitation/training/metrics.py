"""
Unified metrics for imitation learning training and evaluation.

This module provides TorchMetrics-based metrics that work with PyTorch 
Geometric's batch format where features are flattened as [B*N, F] tensors 
(B=num_graphs, N=agents, F=features). The BaseMetric base class extends
MeanMetric to provide automatic averaging and PyG batch utilities.

All metrics integrate seamlessly with PyTorch Lightning's logging system
and can be used directly in LightningModules without a separate collector.
"""
from __future__   import annotations
from torchmetrics import MeanAbsoluteError, MeanMetric, MeanSquaredError
from torchmetrics import MetricCollection, R2Score
from typing       import TYPE_CHECKING

if TYPE_CHECKING:
    from config.imitation.controller  import MurmurationModel, SafetyModel
    from config.imitation.environment import PhysicsModel
    from config.imitation.training    import MetricsModel
    from config.types                 import FlockBatch
    from torch                        import Tensor

import torch as th


class BaseMetric(MeanMetric):
    """
    Base class extending MeanMetric with PyG batch support.
    
    Provides automatic averaging from MeanMetric and PyG batch reshaping helpers.
    Metrics needing state history should use add_state() themselves.
    
    Implementation patterns for metric subclasses:
    
    1. Simple metrics (default): Override evaluate() returning Tensor
       - Always computes a meaningful value from batch
       - BaseMetric.update() calls evaluate() and passes result to MeanMetric
       - Examples: FiedlerValueMetric, SusceptibilityMetric
    
    2. Conditional metrics with zero defaults: Override evaluate() returning Tensor
       - Returns 0 when conditions not met (0 is semantically meaningful)
       - Example: PerturbationResponseMetric (0 = no propagation)
    
    3. Conditional metrics without meaningful defaults: Override update() directly
       - Skip super().update() when computation impossible or invalid
       - Example: ScaleFreeCorrelationMetric (no value when fitting fails)
    
    4. Composite metrics: Override update() and compute() directly
       - Manage multiple sub-metrics internally
       - Return dictionaries from compute()
       - Examples: PowerComponents, States
    """
    _reshape_cache = {}

    def __init__(self, **kwargs):
        """
        Initialize the base metric.
        
        Args:
            **kwargs: Configuration including agent_count and metric-specific params
        """
        super().__init__('ignore')
        self.kwargs = kwargs
    
    def __getattr__(self, name: str):
        """
        Dynamically access config attributes.
        
        Provides access to configuration values passed via kwargs without
        needing to explicitly define each attribute.
        """
        if name in self.kwargs:
            return self.kwargs[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )
    
    def _reshape_features(
        self, 
        batch     : FlockBatch,
        *features : str
    ) -> tuple[Tensor, ...]:
        """
        Reshape flattened PyG features to [B, N, F] format with intelligent caching.
        
        Transforms PyTorch Geometric's flattened tensor format [B*N, F] into the more
        intuitive [B, N, F] shape for batch processing. Features are cached per batch
        to eliminate redundant computations across metrics.
        
        Computed features (lazily evaluated and cached):
        - 'distances' : Pairwise Euclidean distances via cdist      [B, N, N]
        - 'spin_mean' : Mean of normalized velocities across agents [B, 1, 3]
        - 'spins'     : Unit-normalized velocity vectors            [B, N, 3]
        - 'spins_2d'  : Unit-normalized 2D velocity projections     [B, N, 2]
        
        The cache uses batch object IDs as keys, ensuring automatic invalidation when
        processing new batches. The cache is cleared after each training step via
        BaseMetric.clear_cache() to prevent memory growth.
        
        Args:
            batch    : PyG Batch containing flattened features
            features : Variable feature names to retrieve (order preserved)
            
        Returns:
            Tuple of reshaped tensors in requested order, excluding None values
        """
        batch_id  = id(batch)
        B, N      = batch.num_graphs, self.agent_count
        cache     = BaseMetric._reshape_cache 
        normalize = lambda vecs: th.nn.functional.normalize(vecs, dim=-1)
        reshape   = lambda feat: self._reshape_features(batch, feat)[0]
        spin      = lambda s=slice(None): normalize(reshape('velocity')[..., s])
        computed  = {
            'distances'  : lambda: th.cdist(p := reshape('position'), p),
            'spin_mean'  : lambda: spin().mean(dim=1, keepdim=True),
            'spins'      : spin,
            'spins_2d'   : lambda: spin(slice(2)),
        }
        
        get_or_compute = lambda feat: (
            cache[(batch_id, feat)] if (batch_id, feat) in cache
            else 
                (v := (
                    computed[feat]()                if feat in computed
                    else batch[feat].view(B, N, -1) if feat in batch 
                    else None
                ), 
                cache.update({(batch_id, feat): v}), v)[2]
        )
        
        return tuple(
            v for feat in features 
            if (v := get_or_compute(feat)) is not None
        )
    
    @classmethod
    def clear_cache(cls):
        """
        Clear the shared computation cache.
        
        Should be called after each batch to prevent memory growth.
        This would typically be called in Lightning's on_train_batch_end
        and on_validation_batch_end hooks.
        """
        cls._reshape_cache.clear()
    
    def evaluate(self, batch: FlockBatch) -> Tensor:
        """
        Compute metric value from batch.
        
        Subclasses implement this to calculate their specific metric.
        
        Args:
            batch: PyG Batch containing all required data
            
        Returns:
            Computed metric value as a scalar tensor
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement evaluate()"
        )
    
    def update(
        self,
        batch       : FlockBatch,
        predictions : Tensor | None = None,
        targets     : Tensor | None = None
    ):
        """
        Handle MetricCollection's reserved call signature.
        
        Accepts the full signature from MetricCollection but only uses batch.
        Computes metric value via evaluate() and passes to MeanMetric.
        
        Args:
            batch       : PyG Batch containing all required data
            predictions : Predicted actions (unused)
            targets     : Target actions    (unused)
        """
        value = self.evaluate(batch)
        super().update(value)


class FiedlerValueMetric(BaseMetric):
    """
    Measure graph connectivity via the Fiedler value λ₂.

    The algebraic connectivity quantifies how well-connected the flock's
    communication graph is. Higher values indicate stronger cohesion, with
    λ₂ = 0 for disconnected graphs and λ₂ > 0 for connected components.

    The metric computes the second-smallest eigenvalue of the graph Laplacian:

        L = D - A

    where D is the degree matrix and A is the adjacency matrix.
    """

    def evaluate(self, batch: FlockBatch) -> Tensor:
        """
        Compute Fiedler value λ₂ from graph connectivity.
        
        Efficiently computes the second-smallest eigenvalue of the graph
        Laplacian 𝐋 = 𝐃 - 𝐀 using vectorized dense operations.
        
        The spectrum of 𝐋 reveals connectivity properties:

            𝐋𝐯ᵢ = λᵢ𝐯ᵢ with 0 = λ₀ ≤ λ₁ ≤ ... ≤ λₙ₋₁
        
        where λ₁ (the Fiedler value) quantifies algebraic connectivity:
            - λ₁ = 0 for disconnected graphs (multiple components)
            - λ₁ > 0 for connected graphs (single component)
            - λ₁ ∈ (0, 0.1] indicates weak connectivity
            - λ₁ > 0.5 indicates strong cohesion

        Args:
            batch: PyG Batch containing edge_index [2, E] in COO format
            
        Returns:
            Fiedler value as scalar tensor
        """
        device = batch.edge_index.device
        
        A = th.zeros(self.agent_count, self.agent_count, device=device)
        A[batch.edge_index[0], batch.edge_index[1]] = 1.0
        A = (A := A + A.T) - th.diag(A.diagonal())
        
        D = th.diag(degrees) if (degrees := A.sum(dim=1)).any() else th.zeros_like(A)
        L = D - A

        return th.linalg.eigvalsh(L.cpu())[1:2].clamp_min(0.0).to(device)


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
    _hop_cache  = {}
    _triu_cache = {}
    
    @th.compile(mode="default")
    def _compute_hops_per_graph(self, batch: FlockBatch) -> Tensor:
        """
        Compute minimum hop counts using vectorized Floyd-Warshall.
        
        Optimized implementation that processes all graphs in the batch
        simultaneously, eliminating the sequential bottleneck. The k-loop
        now operates on all B graphs at once via broadcasting.
        
        Key optimizations:
            - Vectorized across batch dimension (B×N³ → N³ operations)
            - Single memory allocation with optimal layout
            - Exploits GPU parallelism for via-k path computations
        
        Args:
            batch: PyG Batch with edge_index and batch assignment
            
        Returns:
            Hop distance matrices [B, N, N] computed in parallel
        """
        B, N   = batch.num_graphs, self.agent_count
        device = batch.edge_index.device
        
        hops = th.full(
            device     = device,
            dtype      = th.float32,
            fill_value = float('inf'),
            size       = (B, N, N)
        )
        hops.diagonal(dim1=1, dim2=2).fill_(0)
        
        batch_ids    = batch.batch[batch.edge_index[0]]
        local_source = batch.edge_index[0] - batch_ids * N
        local_target = batch.edge_index[1] - batch_ids * N
        
        hops[batch_ids, local_source, local_target] = 1
        hops[batch_ids, local_target, local_source] = 1
        
        for k in range(N):
            hops = th.minimum(hops, hops[:, :, k:k+1] + hops[:, k:k+1, :])
        
        return hops
    
    def _get_triu_mask(self, device: th.device) -> Tensor:
        """
        Get cached upper triangular mask for symmetric matrix operations.
        
        Caches mask per device to avoid redundant allocations during energy
        computations. The mask ensures each interaction pair counted once.
        """
        cache_key = (device.type, device.index, self.agent_count)
        if cache_key not in self._triu_cache:
            self._triu_cache[cache_key] = th.triu(
                th.ones(self.agent_count, self.agent_count, device=device),
                diagonal = 1
            )
        return self._triu_cache[cache_key]

    def evaluate(self, batch: FlockBatch) -> Tensor:
        """
        Compute Hamiltonian energy using topological interactions.
        
        Computes spin-spin interactions based on hop distances in each graph's
        unique k-NN topology, following Ballerini et al. (2008) findings that
        starling flocks use topological rather than metric interactions.

        Args:
            batch: PyG Batch with edge_index and velocity tensors
            
        Returns:
            Hamiltonian energy values as tensor
        """
        spins,   = self._reshape_features(batch, "spins")
        hops     = self._compute_hops_per_graph(batch)
        coupling = th.where(
            hops.isfinite(),
            self.j_base * (-hops / self.coupling_decay).exp(),
            th.zeros_like(hops)
        )
        
        alert_mod = th.where(
            batch.alert_states.view(-1, self.agent_count, 1) > 0.5,
            self.alert_coupling_factor,
            1.0
        )
        coupling *= alert_mod

        return -(
            coupling
            * th.bmm(spins, spins.mT)
            * self._get_triu_mask(spins.device)
        ).sum(dim=(1, 2)) * 2


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
        
        Tracks edge set evolution via symmetric adjacency matrices to quantify
        topological changes in the flocking graph over time.
        """
        super().__init__(**kwargs)
        self.add_state(
            default = th.zeros(self.agent_count, self.agent_count, dtype=th.bool),
            name    = "last_adjacency"
        )
    
    def evaluate(self, batch: FlockBatch) -> Tensor:
        """
        Measure topological stability via edge set evolution.
        
        Tracks how rapidly neighborhoods reconfigure, distinguishing stable
        cruising (persistent edges) from dynamic maneuvers (edge churn).
        The Jaccard distance quantifies this topological change:
            
            Δ_topo = 1 - |E_t ∩ E_{t-1}| / |E_t ∪ E_{t-1}|
        
        where:
            - E_t : Edge set at time t (undirected)
            - |·| : Cardinality of edge set
            - Δ_topo ∈ [0, 1] : 0 = identical topology, 1 = disjoint
        
        Args:
            batch: PyG Batch containing edge_index [2, E] in COO format
            
        Returns:
            Jaccard distance as scalar tensor
        """
        adjacency = th.zeros(
            self.agent_count * self.agent_count, 
            device = batch.edge_index.device, 
            dtype  = th.bool
        )
        
        flat_indices = th.cat([
            batch.edge_index[0] * self.agent_count + batch.edge_index[1],
            batch.edge_index[1] * self.agent_count + batch.edge_index[0]
        ])
        adjacency[flat_indices] = True
        adjacency = adjacency.view(self.agent_count, self.agent_count)
        triu_idx  = th.triu_indices(
            col    = self.agent_count,
            device = batch.edge_index.device,
            offset = 1, 
            row    = self.agent_count
        )
        
        current_edges       = adjacency[triu_idx[0], triu_idx[1]]
        last_edges          = self.last_adjacency[triu_idx[0], triu_idx[1]]
        self.last_adjacency = adjacency
        intersection        = (current_edges & last_edges).sum()
        union               = (current_edges | last_edges).sum()

        return (
            1.0 - intersection.float() / union.float() 
            if union > 0 else th.tensor(0.0, device=batch.edge_index.device)
        )


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

    def evaluate(self, batch: FlockBatch) -> Tensor:
        """
        Compute polarization measurement.
        
        Uses batched matrix multiplication for efficient computation on
        MPS/GPU, avoiding explicit loops over agent pairs.
        
        Args:
            batch: PyG Batch containing velocity [B*N, 3] flattened
            
        Returns:
            Coherence values as tensor
        """
        headings  = self._reshape_features(batch, "spins_2d")[0]
        alignment = th.bmm(headings, headings.mT)

        return (
            (alignment.sum(dim=(1, 2)) - batch.num_graphs * self.agent_count) / 
            (self.agent_count * (self.agent_count - 1))
        )


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

    def evaluate(self, batch: FlockBatch) -> Tensor:
        """
        Compute wave amplitude measurement.
        
        Uses vectorized distance computations and masked operations for
        efficient gradient calculation on MPS/GPU.
        
        Args:
            batch: PyG Batch with position and velocity [B*N, 3] flattened
            
        Returns:
            Wave amplitude as scalar tensor
        """
        velocities, distances = self._reshape_features(batch, "velocity", "distances")
        headings  = th.atan2(velocities[..., 1], velocities[..., 0])
        mask      = (distances > 0) & (distances < self.orientation_wave_radius)
        
        heading_diffs = (
            lambda h: th.remainder(h + th.pi, 2 * th.pi) - th.pi
        )(headings.unsqueeze(-1) - headings.unsqueeze(-2))
        
        gradients = (
            heading_diffs.abs() / distances.clamp_min(self.epsilon)
        ).masked_fill(~mask, 0)

        return gradients.sum(dim=(1, 2)).mean(dim=0, keepdim=True)


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
        self.add_state("last_velocity", th.empty(0))
        self.add_state("zero_response", th.tensor(0.0))

    def evaluate(self, batch: FlockBatch) -> Tensor:
        """
        Compute threat response measurement.
        
        Measures information propagation efficiency by tracking velocity response
        amplification from threatened to safe agents. Returns 0 when no propagation.
        
        Args:
            batch: PyG Batch with velocity [B*N, 3] and temperature [B*N, 1]
            
        Returns:
            Response ratio as scalar tensor, 0 if no threat propagation
        """
        response_ratio = th.tensor(0.0, device=batch.velocity.device)
        
        if (
            self.last_velocity is not None 
            and batch.velocity.shape == self.last_velocity.shape 
            and (mask := batch.temperature.squeeze(-1) > self.max_temperature).any()
            and (~mask).any()
        ):
            vel_changes     = (batch.velocity - self.last_velocity).norm(dim=-1)
            threat_response = vel_changes[mask].mean().unsqueeze(0)
            
            if (threat_response > self.epsilon).any():
                response_ratio = (
                    vel_changes[~mask].mean().unsqueeze(0) / 
                    threat_response
                )
        
        self.last_velocity = batch.velocity.detach().clone()
        return response_ratio


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
        
        self.forward : MeanMetric = MeanMetric('ignore')
        self.hover   : MeanMetric = MeanMetric('ignore')
        self.lateral : MeanMetric = MeanMetric('ignore')
    
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

    def update(
        self,
        batch       : FlockBatch,
        predictions : Tensor | None = None,
        targets     : Tensor | None = None
    ):
        """
        Update power component measurements with vectorized computation.
        
        Efficiently decomposes control forces using batched operations
        optimized for MPS/GPU execution.
        
        Args:
            batch       : PyG Batch with action [B*N, 3] and velocity [B*N, 3]
            predictions : Predicted actions (unused)
            targets     : Target actions    (unused)
        """
        hover = (batch.action[:, 2] + self.gravity).abs().pow(self.power_exponent)
        self.hover.update(hover)
        
        mask = batch.velocity.norm(dim=-1) > self.velocity_threshold
        if mask.any():
            v_hat     = th.nn.functional.normalize(batch.velocity[mask], dim=-1)
            u_masked  = batch.action[mask]
            forward   = (u_masked * v_hat).sum(dim=-1).clamp_min(0)
            
            forward_power = forward.pow(self.power_exponent)
            lateral_power = (
                u_masked - forward.unsqueeze(-1) * 
                v_hat
            ).norm(dim=-1).pow(self.power_exponent)
            
            full_forward       = th.zeros_like(hover)
            full_lateral       = th.zeros_like(hover)
            full_forward[mask] = forward_power
            full_lateral[mask] = lateral_power
            
            self.forward.update(full_forward)
            self.lateral.update(full_lateral)
        else:
            self.forward.update(th.tensor(0.0))
            self.lateral.update(th.tensor(0.0))


class Regression(BaseMetric):
    """
    Standard regression metrics for action prediction quality.
    
    Tracks multiple loss metrics (MAE, MSE, RMSE, R²) to assess how well
    the learned policy matches expert demonstrations. These metrics provide
    complementary views of prediction error:
    
    - MAE  : Mean absolute error, robust to outliers
    - MSE  : Mean squared error, penalizes large errors  
    - RMSE : Root mean squared error, same units as targets
    - R²   : Coefficient of determination, normalized goodness of fit
    
    All metrics operate on the flattened action space [B*N, 3] matching
    PyTorch Geometric's batch format.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize regression metrics collection.
        
        Creates standard torchmetrics for regression evaluation.
        """
        super().__init__(**kwargs)
        self.mae  = MeanAbsoluteError()
        self.mse  = MeanSquaredError()
        self.r2   = R2Score()
        self.rmse = MeanSquaredError(squared=False)
    
    def compute(self) -> dict[str, Tensor]:
        """
        Compute all regression metrics.
        
        Returns:
            Dictionary with mae, mse, rmse, and r2 scores
        """
        return {
            "mae"  : self.mae.compute(),
            "mse"  : self.mse.compute(),
            "r2"   : self.r2.compute(),
            "rmse" : self.rmse.compute(),
        }
    
    def reset(self):
        """
        Reset all regression metrics.
        """
        self.mae.reset()
        self.mse.reset()
        self.r2.reset()
        self.rmse.reset()
    
    def update(
        self,
        batch       : FlockBatch,
        predictions : Tensor,
        targets     : Tensor
    ):
        """
        Update regression metrics with predictions and targets.
        
        Args:
            batch       : PyG Batch (unused)
            predictions : Predicted actions [B*N, 3]
            targets     : Expert actions    [B*N, 3]
        """
        self.mae.update(predictions,  targets)
        self.mse.update(predictions,  targets)
        self.r2.update(predictions,   targets)
        self.rmse.update(predictions, targets)


class ScaleFreeCorrelationMetric(BaseMetric):
    """
    Measure deviation from scale-free velocity correlations.
    
    Verifies that the flock exhibits power-law velocity correlations
    characteristic of critical systems. The correlation function C(r)
    should follow:

        C(r) ~ r^(-γ)

    where γ ≈ 1/3 for natural murmurations (Cavagna et al. 2010).
    """

    def _bin_correlations(
        self,
        correlations : Tensor,
        distances    : Tensor,
        n_bins       : int
    ) -> tuple[Tensor, Tensor]:
        """
        Bin correlations by distance using logarithmic spacing.
        
        Creates logarithmically-spaced bins to capture scale-free behavior
        across multiple length scales. Uses fully vectorized binning to
        process all batches simultaneously.
        
        Args:
            correlations : Velocity correlations [B_valid, n_pairs]
            distances    : Pairwise distances [B_valid, n_pairs]
            n_bins       : Number of logarithmic bins
            
        Returns:
            Tuple of (bin_means, valid_bins) for power law fitting
        """
        batch     = correlations.shape[0]
        device    = distances.device
        dist_mins = distances.amin(1, keepdim=True)
        dist_maxs = distances.amax(1, keepdim=True)
        
        normalized_bins = (
            (distances.log10() - dist_mins.log10()) / 
            (dist_maxs.log10() - dist_mins.log10() + 1e-8) * n_bins
        ).long().clamp(0, n_bins - 1)
        
        batch_offsets = th.arange(batch, device=device).unsqueeze(1) * n_bins
        bins_flat     = (normalized_bins + batch_offsets).reshape(-1)
        
        bin_sums = th.zeros(batch * n_bins, 2, device=device).index_add_(
            dim    = 0,
            index  = bins_flat,
            source = th.stack(
                dim     = -1,
                tensors = [correlations.reshape(-1), distances.reshape(-1)]
            )
        ).view(batch, n_bins, 2)
        
        counts = th.bincount(
            input     = bins_flat, 
            minlength = batch * n_bins
        ).view(batch, n_bins)
        
        return (
            th.where(
                (valid_bins := counts > 0).unsqueeze(-1),
                bin_sums / counts.clamp_min(1).unsqueeze(-1),
                th.zeros_like(bin_sums)
            ),
            valid_bins
        )

    def _compute_correlations(
        self,
        distances : Tensor,
        spins     : Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Compute velocity correlations for all agent pairs.
        
        Calculates C(r) = ⟨δ𝐯ᵢ · δ𝐯ⱼ⟩ where δ𝐯 = 𝐯 - ⟨𝐯⟩ are
        velocity fluctuations from the mean. Returns upper triangular
        elements only to avoid redundant pair computations.
        
        Args:
            distances : Pairwise distances    [B, N, N]
            spins     : Normalized velocities [B, N, 3]
            
        Returns:
            Tuple of (unique_correlations, unique_distances) [B, N*(N-1)/2]
        """
        triu_idx  = th.triu_indices(self.agent_count, self.agent_count, 1)
        corr_mats = th.bmm(
            delta_spins := spins - spins.mean(dim=1, keepdim=True),
            delta_spins.mT
        )
        
        return (
            corr_mats[:, triu_idx[0], triu_idx[1]], 
            distances[:, triu_idx[0], triu_idx[1]]
        )

    def _fit_power_laws(
        self,
        bin_means  : Tensor,
        valid_bins : Tensor
    ) -> Tensor:
        """
        Fit power law exponents to binned correlation data.
        
        Performs log-log linear regression to estimate γ in C(r) ~ r^(-γ)
        for all valid batches simultaneously. Uses masked operations to
        handle variable numbers of valid bins per batch.
        
        Args:
            bin_means  : Mean correlations and distances per bin [B, n_bins, 2]
            valid_bins : Mask of bins with sufficient data [B, n_bins]
            
        Returns:
            Power law exponents γ for batches with ≥3 valid bins
        """
        log_dists = bin_means[..., 1].log()
        log_corrs = bin_means[..., 0].abs().clamp_min(1e-8).log()
        log_r     = log_dists.masked_fill(~valid_bins, 0)
        log_c     = log_corrs.masked_fill(~valid_bins, 0)
        
        if not (mask := valid_bins.sum(1) >= 3).any():
            return th.empty(0, device=bin_means.device)
        
        X  = log_r[mask] - log_r[mask].mean(1, keepdim=True)
        Y  = log_c[mask] - log_c[mask].mean(1, keepdim=True)
        XX = (X * X * valid_bins[mask]).sum(1)
        XY = (X * Y * valid_bins[mask]).sum(1)
        
        return th.where(XX > 0, -XY / XX, th.zeros_like(XX))

    def update(
        self,
        batch       : FlockBatch,
        predictions : Tensor | None = None,
        targets     : Tensor | None = None
    ):
        """
        Update metric with scale-free correlation measurement.
        
        Override update directly since this metric conditionally computes values
        only when there's sufficient distance variation for power law fitting.
        
        Args:
            batch       : PyG Batch containing position and velocity
            predictions : Predicted actions (unused)
            targets     : Target actions    (unused)
        """
        corrs, dists = self._compute_correlations(
            *self._reshape_features(batch, "distances", "spins")
        )
        
        if not (valid_batch := dists.amax(1) > dists.amin(1)).any():
            return
        
        if (gammas := self._fit_power_laws(
            *self._bin_correlations(
                correlations = corrs[valid_ids := th.where(valid_batch)[0]], 
                distances    = dists[valid_ids], 
                n_bins       = min(10, max(3, corrs.shape[1] // 10))
            )
        )).numel() > 0:
            scaling_deviation = (gammas - self.correlation_exponent).abs()
            super().update(scaling_deviation)


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
        self.acceleration : MeanMetric = MeanMetric('ignore')
        self.temperature  : MeanMetric = MeanMetric('ignore')
        self.velocity     : MeanMetric = MeanMetric('ignore')
    
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

    def update(
        self,
        batch       : FlockBatch,
        predictions : Tensor | None = None,
        targets     : Tensor | None = None
    ):
        """
        Update running sums with batch statistics.
        
        Extracts physical quantities from the batch and accumulates their
        magnitudes for computing running averages across all agents.
        
        Args:
            batch       : PyG Batch containing velocity, temperature, and action
            predictions : Predicted actions (unused)
            targets     : Target actions    (unused)
        """
        self.acceleration.update(batch.action.norm(dim=-1))
        self.temperature.update(batch.temperature)
        self.velocity.update(batch.velocity.norm(dim=-1))




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
        - Dynamic range        (response to weak and strong signals)
        - Information capacity (bandwidth for signal propagation)
        - Correlation length   (scale-free spatial correlations)
    
    Lower susceptibility (χ < 5) indicates an ordered state with reduced
    responsiveness, while very high susceptibility (χ > 20) suggests
    instability. The critical regime χ ∈ [5, 15] balances individual
    freedom with collective coordination.
    """

    def evaluate(self, batch: FlockBatch) -> Tensor:
        """
        Compute susceptibility from velocity fluctuations.
        
        Args:
            batch: PyG Batch containing velocity tensor [B*N, 3] flattened
            
        Returns:
            Susceptibility values as tensor
        """
        spins, spin_mean = self._reshape_features(batch, "spins", "spin_mean")
        polarizations    = (spins * spin_mean).sum(dim=-1)

        return self.agent_count * polarizations.var(dim=-1)


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
            "agent_count"             : agent_count,
            "alert_coupling_factor"   : murmuration.alert_coupling_factor,
            "correlation_exponent"    : metrics.correlation_exponent,
            "coupling_decay"          : murmuration.coupling_decay,
            "epsilon"                 : metrics.epsilon,
            "fiedler_shift"           : metrics.fiedler_shift,
            "gravity"                 : physics.gravity,
            "j_base"                  : murmuration.j_base,
            "max_temperature"         : safety.max_temperature,
            "orientation_wave_radius" : metrics.orientation_wave_radius,
            "power_exponent"          : metrics.power_exponent,
            "power_iterations"        : metrics.power_iterations,
            "velocity_threshold"      : metrics.velocity_threshold,
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
            "neighbor_stability"     : make(NeighborStabilityMetric),
            "orientation_coherence"  : make(OrientationCoherenceMetric),
            "orientation_wave"       : make(OrientationWaveMetric),
            "perturbation_response"  : make(PerturbationResponseMetric),
            "power_components"       : make(PowerComponents),
            "regression"             : make(Regression),
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
