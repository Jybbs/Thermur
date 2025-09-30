"""
Unified metrics for imitation learning training and evaluation.

This module provides TorchMetrics-based metrics that work with PyTorch
Geometric's batch format where features are flattened as [B*N, F] tensors
(B=num_graphs, N=agents, F=features). The BaseMetric base class extends
MeanMetric to provide automatic averaging and PyG batch utilities.

All metrics integrate with PyTorch Lightning's logging system and can be
used directly in LightningModules without a separate collector.
"""
from __future__            import annotations
from torch_geometric.utils import to_dense_adj
from torchmetrics          import MeanAbsoluteError, MeanMetric, MeanSquaredError
from torchmetrics          import Metric, MetricCollection, R2Score
from typing                import TYPE_CHECKING

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
        processing new batches. The cache is cleared after each training frame via
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
        corr      = lambda x: th.bmm(d := x - x.mean(1, keepdim=True), d.mT)
        reshape   = lambda feat: self._reshape_features(batch, feat)[0]
        spins     = lambda: th.nn.functional.normalize(reshape('velocity'), dim=-1)
        triu      = lambda m: m[:, (t := th.triu_indices(N, N, 1))[0], t[1]]
        computed  = {
            'dist_triu'      : lambda: triu(reshape('distances')),
            'spin_corr_triu' : lambda: triu(corr(spins())),
            'spin_mean'      : lambda: spins().mean(dim=1, keepdim=True),
            'spins'          : spins,
            'spins_2d'       : lambda: spins()[..., :2],
            'vel_corr_triu'  : lambda: triu(corr(reshape('velocity'))),
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

    def update(self, batch: FlockBatch, predictions: Tensor):
        """
        Handle MetricCollection's call signature.

        Computes metric value via evaluate() and passes to MeanMetric.

        Args:
            batch       : PyG Batch containing all required data
            predictions : Predicted actions (unused for most metrics)
        """
        value = self.evaluate(batch)
        super().update(value)


class Acceleration(BaseMetric):
    """
    Track average acceleration magnitude across the flock.

    Monitors |𝐚|_avg to quantify control effort and energy expenditure during
    flight maneuvers to ensure physically plausible control forces.

    Tracking this metric ensures the learned policy generates physically
    plausible control forces consistent with bird flight dynamics.
    """

    def evaluate(self, batch: FlockBatch) -> Tensor:
        """
        Compute mean acceleration magnitude.

        Args:
            batch: PyG Batch containing action tensor [B*N, 3]

        Returns:
            Mean acceleration magnitude as scalar tensor
        """
        return batch.action.norm(dim=-1).mean()


class FiedlerValue(BaseMetric):
    """
    Compute Fiedler value λ₂ using adaptive Lanczos iteration.

    Implements the Lanczos algorithm to find the second smallest eigenvalue
    of graph Laplacians, which measures algebraic connectivity. The algorithm
    adaptively determines convergence by monitoring λ₂ stability.

    The Fiedler value λ₂ ∈ [0, ∞) where:
        - λ₂ = 0        : Disconnected graph
        - λ₂ ∈ (0, 0.1] : Weak connectivity
        - λ₂ ∈ (0.1, 1] : Moderate connectivity
        - λ₂ > 1        : Strong connectivity
    """

    def __init__(self, **kwargs):
        """
        Initialize Lanczos Fiedler value computation.

        Args:
            **kwargs: Configuration including agent_count
        """
        super().__init__(**kwargs)
        self.k_used = None

    def _build_laplacian_batch(self, batch: FlockBatch) -> Tensor:
        """
        Construct graph Laplacian matrices for connectivity analysis.

        Builds unnormalized Laplacians L = D - A where D is the degree matrix
        and A the symmetrized binary adjacency. The Laplacian's eigenvalues
        encode graph connectivity properties, with λ₂ (Fiedler value) measuring
        algebraic connectivity.

        Symmetrization ensures undirected edges and binarization removes edge
        weights, critical for consistent Fiedler value computation.

        Args:
            batch: PyG Batch containing edge_index and batch assignment

        Returns:
            Laplacian matrices [B, N, N] for each graph in the batch
        """
        A = (
            adj := to_dense_adj(
                batch         = batch.batch,
                edge_index    = batch.edge_index,
                max_num_nodes = self.agent_count
                )
        ) + adj.mT

        return th.diag_embed(A.sign_().sum(dim=-1)) - A

    def _compute_fiedler(
        self,
        alphas : Tensor,
        betas  : Tensor,
        k      : int
    ) -> Tensor:
        """
        Compute Fiedler values from Lanczos coefficients via tridiagonal reduction.

        Constructs symmetric tridiagonal matrix T from Lanczos coefficients that
        preserves the spectral properties of the original N×N Laplacian. The k×k
        tridiagonal approximation T ≈ Q^T L Q contains the same eigenvalue
        information as the full Laplacian but in a compact form suitable for
        efficient eigendecomposition.

        Building and computing on CPU avoids repeated device transfers and leverages
        optimized LAPACK routines for symmetric tridiagonal eigenvalue problems.

        Args:
            alphas : Diagonal Lanczos coefficients [B, :] from orthogonal projections
            betas  : Off-diagonal coupling terms [B, :] encoding subspace recurrence
            k      : Number of Lanczos iterations to include (matrix dimension)

        Returns:
            Fiedler values λ₂ (second smallest eigenvalues) for each graph [B]
        """
        alphas_k  = alphas[:, :k].cpu()
        diagonals = th.diag_embed(alphas_k)

        if k > 1:
            off_diagonals = betas[:, :k - 1].cpu()
            diagonals[:, range(k - 1), range(1,  k)] = off_diagonals
            diagonals[:, range(1,  k), range(k - 1)] = off_diagonals

        return th.linalg.eigvalsh(diagonals)[:, 0]

    def _compute_lanczos_eigenvalue(self, laplacians: Tensor) -> Tensor:
        """
        Compute Fiedler values λ₂ via adaptive Lanczos iteration.

        Iteratively constructs tridiagonal approximation T ≈ Q^T L Q using
        the Lanczos algorithm, where T captures the essential spectral properties
        of L in a smaller k×k matrix. The algorithm adaptively determines k by
        monitoring convergence of λ₂.

        Early termination occurs when eigenvalue stability is achieved, typically
        requiring only 20-50% of full iterations for well-connected graphs.

        Args:
            laplacians: Batch of Laplacian matrices [B, N, N]

        Returns:
            Fiedler values λ₂ for each graph [B]
        """
        B, N, _ = laplacians.shape
        device  = laplacians.device
        alphas  = th.zeros(B, self.agent_count,     device=device)
        betas   = th.zeros(B, self.agent_count - 1, device=device)

        v_init       = th.randn(B, N, device=device)
        v_init      -= v_init.mean(dim=1, keepdim=True)
        v_curr       = v_init / v_init.norm(dim=1, keepdim=True)
        v_prev       = th.zeros_like(v_curr)
        prev_fiedler = th.zeros(B)

        for i in range(self.agent_count):
            w     = th.einsum('bij, bj -> bi', laplacians, v_curr)
            alpha = th.einsum('bi,  bi -> b', w, v_curr)
            alphas[:, i] = alpha

            w -= alpha[:, None] * v_curr + betas[:, max(0, i-1), None] * v_prev
            w -= w.mean(dim=1, keepdim=True)

            if (beta := w.norm(dim=1)).min() <= 1e-10 or i >= self.agent_count - 1:
                break

            betas[:, i] = beta
            v_prev = v_curr
            v_curr = w / beta[:, None]

            if i >= 10 and i % 5 == 0:
                if th.allclose(
                    curr_fiedler := self._compute_fiedler(alphas, betas, (k := i + 1)),
                    prev_fiedler
                ):
                    self.k_used = k
                    return curr_fiedler.to(device)
                prev_fiedler = curr_fiedler

        self.k_used = i + 1
        return self._compute_fiedler(alphas, betas, self.k_used).to(device)

    def evaluate(self, batch: FlockBatch) -> Tensor:
        """
        Compute harmonic mean of Fiedler values.

        The harmonic mean H = n / Σ(1/λᵢ) emphasizes weak connectivity,
        making it sensitive to poorly connected components where a single
        disconnected subgroup represents system failure.

        Args:
            batch: PyG Batch with edge_index and graph assignments

        Returns:
            Harmonic mean of Fiedler values as scalar tensor
        """
        laplacians = self._build_laplacian_batch(batch)
        fiedler    = self._compute_lanczos_eigenvalue(laplacians).clamp_min(1e-10)
        return len(fiedler) / fiedler.reciprocal().sum()


class MAE(Metric):
    """
    Mean Absolute Error for action predictions.

    Measures the average absolute difference between predicted and expert
    actions in m/s² units. MAE is more robust to outliers than MSE/RMSE,
    providing a complementary view of prediction accuracy.

    Performance interpretation:
        - MAE < 1.0 m/s²  : Excellent - near-expert performance
        - MAE ∈ [1, 2]    : Good      - effective imitation
        - MAE ∈ [2, 4]    : Moderate  - functional but imprecise
        - MAE > 4.0 m/s²  : Poor      - significant prediction errors
    """

    def __init__(self, **kwargs):
        """
        Initialize MAE metric.

        Args:
            **kwargs: Unused, accepted for factory compatibility
        """
        super().__init__()
        self.metric = MeanAbsoluteError()

    def compute(self) -> th.Tensor:
        """
        Compute the metric value.

        Returns:
            Computed MAE value as scalar tensor
        """
        return self.metric.compute()

    def reset(self):
        """
        Reset metric state.
        """
        self.metric.reset()

    def update(self, batch: FlockBatch, predictions: Tensor):
        """
        Update metric with predictions and targets.

        Args:
            batch       : PyG Batch containing target actions
            predictions : Predicted actions from the policy
        """
        self.metric.update(predictions, batch.action)


class MaxEntropyEnergy(BaseMetric):
    """
    Track effective energy E = -Σ_{⟨ij⟩} J_{ij} 𝐬ᵢ·𝐬ⱼ per frame.

    Computes the interaction energy from the maximum entropy formulation
    following Bialek et al. (2012), where velocities act as spin variables:

        E = -Σᵢⱼ J_{ij} (v̂ᵢ · v̂ⱼ)

    where J_{ij} = J₀ exp(-dᵢⱼ/λ) with:
        - J₀ is the uniform base coupling strength
        - dᵢⱼ is topological distance from k-NN graph
        - λ is the coupling decay length

    This energy function emerges from statistical inference rather than
    mechanics, with heterogeneous noise creating critical state variance.
    """
    _hop_cache  = {}
    _triu_cache = {}

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
        Compute effective energy using topological interactions.

        Computes spin-spin interactions based on hop distances in each graph's
        unique k-NN topology, following Ballerini et al. (2008) findings that
        starling flocks use topological rather than metric interactions.

        Args:
            batch: PyG Batch with edge_index and velocity tensors

        Returns:
            Effective energy values as tensor
        """
        spins = self._reshape_features(batch, "spins")[0]
        hops  = self._compute_hops_per_graph(batch)

        coupling = th.where(
            hops.isfinite(),
            self.j_base * (-hops / self.coupling_decay).exp(),
            th.zeros_like(hops)
        )

        return -(
            coupling
            * th.bmm(spins, spins.mT)
            * self._get_triu_mask(spins.device)
        ).sum(dim=(1, 2)) * 2


class NeighborStability(BaseMetric):
    """
    Quantify topological stability of the communication graph.

    Measures the Jaccard distance between consecutive graph states to
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


class NoiseHeterogeneity(BaseMetric):
    """
    Monitor heterogeneity variance to ensure critical state maintenance.

    Tracks the standard deviation of individual noise amplitudes η_i across
    the flock, where heterogeneity creates the behavioral variance necessary
    for murmuration patterns. From Guisandez et al. (2018), maintaining
    σ(η) at the configured heterogeneity_std ensures continuous phase
    transitions and scale-free correlations.

    The metric computes:
        Δσ = |σ(η_batch) - σ_target|

    where σ(η_batch) is the measured standard deviation of noise amplitudes
    and σ_target is the configured heterogeneity_std parameter.
    """

    def evaluate(self, batch: FlockBatch) -> Tensor:
        """
        Measure deviation from target heterogeneity variance.

        Args:
            batch: PyG Batch containing heterogeneity values

        Returns:
            Mean absolute deviation from target σ across batch
        """
        return (
            batch.heterogeneity.view(batch.num_graphs, self.agent_count)
                .std(dim=1)
                .sub(self.heterogeneity_std)
                .abs()
                .mean()
        )


class OrientationCoherence(BaseMetric):
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
            Mean coherence across batch as scalar tensor
        """
        headings  = self._reshape_features(batch, "spins_2d")[0]
        alignment = th.bmm(headings, headings.mT)

        return (
            (alignment.sum(dim=(1, 2)) - self.agent_count) /
            (self.agent_count * (self.agent_count - 1))
        ).mean()


class OrientationWave(BaseMetric):
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
            heading_diffs.abs() / distances.clamp_min(1e-3)
        ).masked_fill(~mask, 0)

        return gradients.sum(dim=(1, 2)).mean(dim=0, keepdim=True)


class PerturbationResponse(BaseMetric):
    """
    Quantify collective response to thermal perturbations χ_thermal.

    Measures information propagation efficiency by tracking velocity response
    amplification from threatened to safe agents:

        χ_thermal = ⟨|Δ𝐯_safe|⟩ / ⟨|Δ𝐯_threat|⟩

    where Δ𝐯 = 𝐯(t) - 𝐯(t-Δt) represents velocity changes between frames.

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

            if (threat_response > 1e-3).any():
                response_ratio = (
                    vel_changes[~mask].mean().unsqueeze(0) /
                    threat_response
                )

        self.last_velocity = batch.velocity.detach().clone()
        return response_ratio


class R2(Metric):
    """
    R-squared score for action predictions.

    Measures the proportion of variance in expert actions explained by the
    policy's predictions. R² ∈ [−∞, 1] where 1 indicates perfect prediction,
    0 indicates performance equivalent to predicting the mean, and negative
    values indicate worse than mean prediction.

    Performance interpretation:
        - R² > 0.90      : Excellent - captures most variance
        - R² ∈ [0.7, 0.9]: Good      - strong predictive power
        - R² ∈ [0.3, 0.7]: Moderate  - partial pattern learning
        - R² < 0.30      : Poor      - minimal predictive capability
        - R² < 0         : Failure   - worse than mean prediction
    """

    def __init__(self, **kwargs):
        """
        Initialize R² score metric.

        Args:
            **kwargs: Unused, accepted for factory compatibility
        """
        super().__init__()
        self.metric = R2Score()

    def compute(self) -> th.Tensor:
        """
        Compute the metric value.

        Returns:
            Computed R² value as scalar tensor
        """
        return self.metric.compute()

    def reset(self):
        """
        Reset metric state.
        """
        self.metric.reset()

    def update(self, batch: FlockBatch, predictions: Tensor):
        """
        Update metric with predictions and targets.

        Args:
            batch       : PyG Batch containing target actions
            predictions : Predicted actions from the policy
        """
        self.metric.update(predictions, batch.action)


class RMSE(Metric):
    """
    Root Mean Squared Error for action predictions.

    Measures the typical prediction error in m/s² units, giving more weight
    to large errors than MAE. RMSE provides an interpretable metric in the
    same units as the predictions, making it ideal for monitoring training.

    Performance interpretation:
        - RMSE < 1.5 m/s² : Excellent - high-fidelity imitation
        - RMSE ∈ [1.5, 3] : Good      - acceptable prediction errors
        - RMSE ∈ [3, 5]   : Moderate  - noticeable deviations
        - RMSE > 5.0 m/s² : Poor      - large prediction errors

    Note that RMSE ≥ MAE with equality only when all errors are identical.
    """

    def __init__(self, **kwargs):
        """
        Configures MeanSquaredError with squared=False to compute root mean
        squared error rather than mean squared error.

        Args:
            **kwargs: Unused, accepted for factory compatibility
        """
        super().__init__()
        self.metric = MeanSquaredError(squared=False)

    def compute(self) -> th.Tensor:
        """
        Compute the metric value.

        Returns:
            Computed RMSE value as scalar tensor
        """
        return self.metric.compute()

    def reset(self):
        """
        Reset metric state.
        """
        self.metric.reset()

    def update(self, batch: FlockBatch, predictions: Tensor):
        """
        Update metric with predictions and targets.

        Args:
            batch       : PyG Batch containing target actions
            predictions : Predicted actions from the policy
        """
        self.metric.update(predictions, batch.action)


class ScaleFreeCorrelation(BaseMetric):
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
            valid_bins : Mask of bins with sufficient data       [B, n_bins]

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

    def update(self, batch: FlockBatch, predictions: Tensor):
        """
        Update metric with scale-free correlation measurement.

        Override update directly since this metric conditionally computes values
        only when there's sufficient distance variation for power law fitting.

        Args:
            batch       : PyG Batch containing position and velocity
            predictions : Predicted actions (unused)
        """
        corrs, dists = self._reshape_features(batch, "spin_corr_triu", "dist_triu")

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
            MeanMetric.update(self, scaling_deviation)


class Susceptibility(BaseMetric):
    """
    Measures flock susceptibility as integrated velocity correlations.

    Following Cavagna et al. (2010) and Attanasi et al. (2014), susceptibility
    quantifies the total correlation in the system through the cumulative
    correlation function:

        Q(r) = ∫₀ʳ C(r')dr'

    where C(r) is the velocity correlation function:

        C(r) = ⟨δ𝐯ᵢ · δ𝐯ⱼ⟩ for pairs at distance r

    with velocity fluctuations δ𝐯ᵢ = 𝐯ᵢ - (1/N)Σₖ𝐯ₖ.

    The susceptibility χ = Q(ξ) is the maximum of the cumulative correlation,
    reached at the correlation length ξ where C(ξ) = 0.

    In critical systems, χ scales with flock size N without saturation,
    indicating maintained responsiveness at all scales. The ratio χ/N
    remains finite, characteristic of near-critical dynamics that enable
    rapid information transfer across the entire flock.
    """

    def __init__(self, **kwargs):
        """
        Initialize susceptibility metric with adaptive binning.

        Following empirical analysis in scale-free systems (Cavagna et al. 2010),
        the number of bins scales as O(√N) to balance resolution with statistics.
        This ensures adequate sampling while maintaining computational efficiency.
        """
        super().__init__(**kwargs)

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
            distances    : Pairwise distances    [B_valid, n_pairs]
            n_bins       : Number of logarithmic bins

        Returns:
            Tuple of (bin_means, valid_bins) for integration
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

    def _integrate_cumulative(
        self,
        bin_means  : Tensor,
        valid_bins : Tensor
    ) -> Tensor:
        """
        Integrate correlation function using trapezoidal rule.

        Computes cumulative correlation Q(r) = ∫₀ʳ C(r')dr' via numerical
        integration. The susceptibility χ is the maximum value of Q(r),
        typically reached at the correlation length where C(r) → 0.

        Args:
            bin_means  : Mean correlations and distances per bin [B, n_bins, 2]
            valid_bins : Mask of bins with sufficient data       [B, n_bins]

        Returns:
            Susceptibility χ as maximum integrated correlation for batches
            with at least 2 consecutive valid bins, empty tensor otherwise
        """
        consecutive = valid_bins[:, :-1] & valid_bins[:, 1:]
        if not (mask := consecutive.any(1)).any():
            return th.empty(0, device=bin_means.device)

        correlations = bin_means[..., 0]
        distances    = bin_means[..., 1]
        cumulative   = th.zeros_like(correlations)

        for i in range(1, bin_means.shape[1]):
            both_valid = valid_bins[:, i] & valid_bins[:, i-1]
            delta_r    = (distances[:, i] - distances[:, i-1]).abs()
            trapz_area = 0.5 * (correlations[:, i] + correlations[:, i-1]) * delta_r

            cumulative[:, i] = th.where(
                both_valid,
                cumulative[:, i-1] + trapz_area,
                cumulative[:, i-1]
            )

        return cumulative[mask].abs().amax(dim=1)

    def update(self, batch: FlockBatch, predictions: Tensor):
        """
        Update metric with susceptibility measurement.

        Override update directly since this metric conditionally computes values
        only when there's sufficient data for meaningful integration.

        Args:
            batch       : PyG Batch containing position and velocity
            predictions : Predicted actions (unused)
        """
        corrs, dists = self._reshape_features(batch, "vel_corr_triu", "dist_triu")

        if not (valid_batch := dists.amax(1) > dists.amin(1)).any():
            return

        if (susceptibilities := self._integrate_cumulative(
            *self._bin_correlations(
                correlations = corrs[valid_ids := th.where(valid_batch)[0]],
                distances    = dists[valid_ids],
                n_bins       = min(20, max(5, corrs.shape[1] // 10))
            )
        )).numel() > 0:
            MeanMetric.update(self, susceptibilities)


class Temperature(BaseMetric):
    """
    Track average temperature sensed across the flock.

    Monitors mean temperature θ_avg to assess thermal threat exposure and
    safety constraint satisfaction. Temperature serves as the primary signal
    for environmental hazards in the murmuration model, with agents detecting
    and propagating threat information through the flock.

    This metric verifies that the learned policy maintains appropriate thermal
    awareness and triggers collective evasion when temperature thresholds are
    exceeded.
    """

    def evaluate(self, batch: FlockBatch) -> Tensor:
        """
        Compute mean temperature.

        Args:
            batch: PyG Batch containing temperature tensor [B*N, 1]

        Returns:
            Mean temperature as scalar tensor
        """
        return batch.temperature.mean()


class Velocity(BaseMetric):
    """
    Track average velocity magnitude across the flock.

    Monitors |𝐯|_avg to ensure realistic flight speeds are maintained during
    collective motion. Empirical observations from Ballerini et al. (2008) show
    starlings cruise at approximately 11.1 m/s, with speeds ranging from 9-12 m/s
    during murmuration events.

    This metric validates that the learned policy preserves the characteristic
    speed regulation of self-propelled particles in active matter systems,
    where agents maintain preferred speeds despite interactions and perturbations.
    """

    def evaluate(self, batch: FlockBatch) -> Tensor:
        """
        Compute mean velocity magnitude.

        Args:
            batch: PyG Batch containing velocity tensor [B*N, 3]

        Returns:
            Mean velocity magnitude as scalar tensor
        """
        return batch.velocity.norm(dim=-1).mean()


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
            "correlation_exponent"    : metrics.correlation_exponent,
            "coupling_decay"          : murmuration.coupling_decay,
            "fiedler_shift"           : metrics.fiedler_shift,
            "gravity"                 : physics.gravity,
            "heterogeneity_std"       : murmuration.heterogeneity_std,
            "j_base"                  : murmuration.j_base,
            "max_temperature"         : safety.max_temperature,
            "orientation_wave_radius" : metrics.orientation_wave_radius,
            "velocity_threshold"      : metrics.velocity_threshold,
        }

    def create_training_metrics(self) -> MetricCollection:
        """
        Create complete training metric collection.

        Returns:
            MetricCollection with all configured metrics for training
        """
        make = lambda cls: cls(**self.cfg)

        return MetricCollection(
            metrics = {
                "acceleration"           : make(Acceleration),
                "fiedler_value"          : make(FiedlerValue),
                "max_entropy_energy"     : make(MaxEntropyEnergy),
                "mae"                    : make(MAE),
                "neighbor_stability"     : make(NeighborStability),
                "noise_heterogeneity"    : make(NoiseHeterogeneity),
                "orientation_coherence"  : make(OrientationCoherence),
                "orientation_wave"       : make(OrientationWave),
                "perturbation_response"  : make(PerturbationResponse),
                "r2"                     : make(R2),
                "rmse"                   : make(RMSE),
                "scale_free_correlation" : make(ScaleFreeCorrelation),
                "susceptibility"         : make(Susceptibility),
                "temperature"            : make(Temperature),
                "velocity"               : make(Velocity),
            },
            prefix = "training/"
        )

    def create_validation_metrics(self) -> MetricCollection:
        """
        Clone training metrics with validation prefix.

        Returns:
            MetricCollection with all configured metrics for validation
        """
        return self.create_training_metrics().clone(prefix="validation/")
