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
    Subclasses override evaluate() to return a scalar Tensor computed from the batch.
    The update() method automatically calls evaluate() and passes the result to
    MeanMetric for averaging.
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

        The cache uses batch object IDs as keys, ensuring automatic invalidation when
        processing new batches. The cache is cleared after each training frame via
        BaseMetric.clear_cache() to prevent memory growth.

        Args:
            batch    : PyG Batch containing flattened features
            features : Variable feature names to retrieve (order preserved)

        Returns:
            Tuple of reshaped tensors in requested order, excluding None values
        """
        b_id       = id(batch)
        B, E, N    = batch.num_graphs, batch.edge_index, self.agent_count
        cache      = BaseMetric._reshape_cache
        adj        = lambda   : to_dense_adj(E, batch.batch, max_num_nodes=N)
        fluct      = lambda   : (v := reshape('velocity')) - v.mean(dim=1, keepdim=True)
        norm       = lambda f : f.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        reshape    = lambda f : self._reshape_features(batch, f)[0]
        spins      = lambda   : th.nn.functional.normalize(reshape('velocity'), dim=-1)

        computed = {
            'adjacency'  : lambda: ((A := adj()) + A.mT).sign(),
            'fluct_corr' : lambda: th.bmm(fn := (f := fluct()) / norm(f), fn.mT),
            'spins'      : spins,
            'spins_2d'   : lambda: spins()[..., :2],
            'vel_mag'    : lambda: reshape('velocity').norm(dim=-1),
        }

        get_or_compute = lambda feat: (
            cache[(b_id, feat)] if (b_id, feat) in cache
            else
                (v := (
                    computed[feat]()                if feat in computed
                    else batch[feat].view(B, N, -1) if feat in batch
                    else None
                ),
                cache.update({(b_id, feat): v}), v)[2]
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


class ClusteringCoefficient(BaseMetric):
    """
    Measure local neighborhood cohesion via clustering coefficient.

    Quantifies the degree to which nodes cluster together by computing the
    ratio of closed triangles to possible triangles in each node's neighborhood.
    The clustering coefficient C ranges from 0 to 1:

        C = (3 × number of triangles) / (number of connected triples)

    Higher values (C → 1) indicate stable, cohesive local neighborhoods where
    an agent's neighbors are also connected to each other. Lower values (C → 0)
    suggest sparse or reconfiguring topologies typical of dynamic maneuvers.

    This instantaneous metric captures topological stability without requiring
    temporal comparisons, making it suitable for shuffled training data.

    Expected ranges:
        - Stable cruising    : C ∈ [0.6, 1.0]
        - Active maneuvering : C ∈ [0.3, 0.6]
        - Sparse/fragmented  : C ∈ [0.0, 0.3]
    """

    def evaluate(self, batch: FlockBatch) -> Tensor:
        """
        Compute average clustering coefficient across all graphs.

        Uses batched dense adjacency matrices for vectorized triangle counting.
        For each node, computes the fraction of its neighbors that are also
        connected to each other, then averages across all nodes and graphs.

        The computation follows:
            1. Convert edge_index to symmetrized dense adjacency [B, N, N]
            2. Binarize to {0, 1} since symmetrization can create values of 2
            3. Count triangles via A ⊙ A² where ⊙ is element-wise product
            4. Compute C_i = triangles_i / (k_i * (k_i - 1)) for degree k_i

        Args:
            batch: PyG Batch containing edge_index [2, E] and batch assignment

        Returns:
            Mean clustering coefficient as scalar tensor
        """
        A, = self._reshape_features(batch, 'adjacency')

        degree         = A.sum(dim=2)
        triangles      = (A @ A * A).sum(dim=2)
        possible_pairs = degree * (degree - 1)

        valid_mask = possible_pairs > 0
        clustering = th.zeros_like(degree, dtype=th.float)
        clustering[valid_mask] = (
            triangles[valid_mask] / possible_pairs[valid_mask].float()
        )

        return clustering.mean()


class CorrelationLength(BaseMetric):
    """
    Measure normalized correlation length ξ/L as scale-free indicator.

    Following Cavagna et al. (2010), the correlation length ξ characterizes
    the spatial scale over which velocity fluctuations remain correlated:

        C(r) = ⟨δφ̃ᵢ · δφ̃ⱼ⟩  for |rᵢ - rⱼ| = r

    where δφ̃ᵢ are normalized velocity fluctuations. The correlation length ξ
    is defined as the distance where C(r) decays to a threshold (typically 0.6).

    The normalized ratio ξ/L (where L is system size) indicates criticality:

        ξ/L ≈ 0.35 : Near-critical (scale-free correlations)
        ξ/L < 0.2  : Subcritical (short-range order)
        ξ/L > 0.5  : Supercritical (over-correlated)
    """

    def _bin_correlations(
        self,
        correlations : Tensor,
        distances    : Tensor,
        n_bins       : int
    ) -> tuple[Tensor, Tensor]:
        """
        Bin correlations by distance using logarithmic spacing.

        Creates logarithmically-spaced bins to capture correlation decay
        across multiple length scales. Uses fully vectorized binning to
        process all batches simultaneously.

        Args:
            correlations : Velocity fluctuation correlations [B, N, N]
            distances    : Pairwise distances                [B, N, N]
            n_bins       : Number of logarithmic bins

        Returns:
            Tuple of (bin_distances, bin_correlations) averaged per bin [B, n_bins]
        """
        batch     = correlations.shape[0]
        device    = distances.device
        dist_mins = distances.amin(dim=(1, 2), keepdim=True)
        dist_maxs = distances.amax(dim=(1, 2), keepdim=True)

        normalized_bins = (
            (distances.log10() - dist_mins.log10()) /
            (dist_maxs.log10() - dist_mins.log10() + 1e-8) * n_bins
        ).long().clamp(0, n_bins - 1)

        batch_offsets = th.arange(batch, device=device).view(batch, 1, 1) * n_bins
        bins_flat     = (normalized_bins + batch_offsets).reshape(-1)

        bin_sums = th.zeros(batch * n_bins, 2, device=device)
        bin_sums.scatter_add_(
            dim   = 0,
            index = bins_flat.unsqueeze(-1).expand(-1, 2),
            src   = th.stack(
                dim     = -1,
                tensors = [distances.reshape(-1), correlations.reshape(-1)]
            )
        )
        bin_sums = bin_sums.view(batch, n_bins, 2)

        counts = th.bincount(
            input     = bins_flat,
            minlength = batch * n_bins
        ).view(batch, n_bins).clamp_min(1)

        return (
            bin_sums[..., 0] / counts, 
            bin_sums[..., 1] / counts
        )

    def _find_crossing(
        self,
        bin_distances    : Tensor,
        bin_correlations : Tensor,
        threshold        : float
    ) -> Tensor:
        """
        Find distance ξ where C(r) crosses threshold via linear interpolation.

        For bins where C(rᵢ) ≥ θ > C(rᵢ₊₁), linearly interpolate:

            ξ = rᵢ + (θ - C(rᵢ))/(C(rᵢ₊₁) - C(rᵢ)) · (rᵢ₊₁ - rᵢ)

        where θ is the correlation threshold and i is the first bin below threshold.

        Args:
            bin_distances    : Mean distance per bin    [B, n_bins]
            bin_correlations : Mean correlation per bin [B, n_bins]
            threshold        : Correlation threshold θ to detect crossing

        Returns:
            Correlation length ξ for each batch [B]
        """
        below_threshold = bin_correlations < threshold
        first_below     = below_threshold.int().argmax(dim=1)
        bracket_indices = th.stack([(
            first_below - 1).clamp_min(0), 
            first_below
        ], dim=-1)

        bracket_dists = bin_distances.gather(1, bracket_indices)
        bracket_corrs = bin_correlations.gather(1, bracket_indices)
        interpolation = (
            (threshold - bracket_corrs[:, 0]) /
            (bracket_corrs[:, 1] - bracket_corrs[:, 0] + 1e-8)
        )
        
        correlation_length = (
            bracket_dists[:, 0] +
            interpolation * (bracket_dists[:, 1] - bracket_dists[:, 0])
        )

        return th.where(
            (first_below > 0) & below_threshold.any(dim=1),
            correlation_length,
            th.zeros_like(correlation_length)
        )

    def evaluate(self, batch: FlockBatch) -> Tensor:
        """
        Compute normalized correlation length ξ/L.

        Args:
            batch: PyG Batch containing positions [B*N, 3] and velocities [B*N, 3]

        Returns:
            Mean ξ/L across batch as scalar tensor
        """
        correlations, = self._reshape_features(batch, 'fluct_corr')
        distances,    = self._reshape_features(batch, 'distances')
        positions,    = self._reshape_features(batch, 'position')

        correlation_length = self._find_crossing(
            *self._bin_correlations(
                correlations = correlations, 
                distances    = distances, 
                n_bins       = min(20, max(5, self.agent_count // 10))
            ),
            threshold = self.correlation_threshold
        )

        return (
            correlation_length / 
            positions.std(dim=1).norm(dim=-1).clamp_min(1e-8)
        ).mean()


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

        Args:
            batch: PyG Batch containing edge_index and batch assignment

        Returns:
            Laplacian matrices [B, N, N] for each graph in the batch
        """
        A, = self._reshape_features(batch, 'adjacency')
        return th.diag_embed(A.sum(dim=-1)) - A

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

        The energy is calculated as the sum of pairwise couplings for each
        agent, then averaged across agents within each batch, providing a
        mean energy per agent that is independent of flock size.

        Args:
            batch: PyG Batch with edge_index and velocity tensors

        Returns:
            Mean energy per agent as scalar tensor
        """
        spins    = self._reshape_features(batch, 'spins')[0]
        hops     = self._compute_hops_per_graph(batch)
        coupling = th.where(
            hops.isfinite(),
            self.j_base * (-hops / self.coupling_decay).exp(),
            th.zeros_like(hops)
        )

        pairwise_energies = -(
            coupling
            * th.bmm(spins, spins.mT)
            * self._get_triu_mask(spins.device)
        ) * 2

        return (
            pairwise_energies.sum(dim=-1)
                .mean(dim=1, keepdim=True)
                .mean(dim=0, keepdim=True)
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

        Computes the mean local gradient of heading angles |dθ/dr|
        averaged first over each agent's neighbors within radius R_wave,
        then across all agents in the flock. This properly normalizes
        the metric to be independent of neighbor count.

        Uses vectorized distance computations and masked operations for
        efficient gradient calculation on MPS/GPU.

        Args:
            batch: PyG Batch with position and velocity [B*N, 3] flattened

        Returns:
            Wave amplitude as scalar tensor (rad/m)
        """
        dists, vels = self._reshape_features(batch, 'distances', 'velocities')
        headings    = th.atan2(vels[..., 1], vels[..., 0])
        mask        = (dists > 0) & (dists < self.orientation_wave_radius)

        heading_diffs = (
            lambda h: th.remainder(h + th.pi, 2 * th.pi) - th.pi
        )(headings.unsqueeze(-1) - headings.unsqueeze(-2))

        gradients = (
            heading_diffs.abs() / dists.clamp_min(1e-3)
        ).masked_fill(~mask, 0)

        valid_neighbors     = mask.sum(dim=-1).clamp_min(1)
        mean_grad_per_agent = gradients.sum(dim=-1) / valid_neighbors

        return (
            mean_grad_per_agent
                .mean(dim=1, keepdim=True)
                .mean(dim=0, keepdim=True)
        )


class PairwiseCoherence(BaseMetric):
    """
    Measure mean pairwise velocity alignment across local neighborhoods.

    Computes the average dot product of normalized velocity pairs, capturing
    local coordination quality:

        C = ⟨v̂ᵢ · v̂ⱼ⟩_{i≠j} = (1/[N(N-1)]) Σᵢ≠ⱼ v̂ᵢ · v̂ⱼ

    where v̂ᵢ = vᵢ/|vᵢ| are normalized velocity vectors. This metric quantifies
    whether nearby agents are aligned with each other, independent of whether
    they share a common global direction.

    Pairwise coherence complements polarization by detecting local coordination
    during maneuvers where global direction changes (e.g., coordinated turns,
    vortex formations) but local alignment remains high.

    Expected ranges:
        Random motion       : C ≈ 0
        Loose aggregation   : C ∈ [0.1, 0.3]
        Coordinated turning : C ∈ [0.3, 0.6]
        Aligned cruise      : C ∈ [0.6, 1.0]
    """

    def evaluate(self, batch: FlockBatch) -> Tensor:
        """
        Compute mean pairwise alignment via batched matrix multiplication.

        Args:
            batch: PyG Batch containing velocity [B*N, 3]

        Returns:
            Mean pairwise coherence across batch as scalar tensor
        """
        spins,    = self._reshape_features(batch, 'spins_2d')[0]
        alignment = th.bmm(spins, spins.mT)

        return (
            (alignment.sum(dim=(1, 2)) - self.agent_count) /
            (self.agent_count * (self.agent_count - 1))
        ).mean()


class Polarization(BaseMetric):
    """
    Measure global alignment through polarization order parameter Φ.

    Following Cavagna et al. (2010), polarization quantifies collective
    alignment of velocity directions:

        Φ = |⟨v̂ᵢ⟩| = |(1/N) Σᵢ v̂ᵢ|

    where v̂ᵢ = vᵢ/|vᵢ| are normalized velocity vectors. This captures the
    phase transition between disordered (Φ ≈ 0) and ordered (Φ ≈ 1) states.

    Natural murmurations exhibit high polarization Φ ≈ 0.96 ± 0.03, indicating
    strong directional alignment while maintaining velocity fluctuations that
    enable collective response to perturbations.

    Expected ranges (Cavagna et al. 2010):
        Random motion     : Φ ∈ [0.0, 0.2]
        Loose aggregation : Φ ∈ [0.2, 0.5]
        Coordinated turn  : Φ ∈ [0.5, 0.8]
        Murmuration       : Φ ∈ [0.90, 0.97]
        Over-synchronized : Φ > 0.97 (pathological)
    """

    def evaluate(self, batch: FlockBatch) -> Tensor:
        """
        Compute polarization as magnitude of mean normalized velocity.

        Args:
            batch: PyG Batch containing velocity [B*N, 3]

        Returns:
            Mean polarization across batch as scalar tensor
        """
        spins, = self._reshape_features(batch, 'spins_2d')
        return spins.mean(dim=1).norm(dim=-1).mean()


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


class Susceptibility(BaseMetric):
    """
    Measure per-agent susceptibility χ/N as normalized collective response.

    Following Attanasi et al. (2014), susceptibility per agent quantifies the
    system's collective response normalized by flock size:

        χ/N = (1/N²) Σᵢ≠ⱼ ⟨δφ̃ᵢ · δφ̃ⱼ⟩ θ(r₀ - rᵢⱼ)

    where:
        δφ̃ᵢ = (δvᵢ/|δvᵢ|) are normalized velocity fluctuations
        δvᵢ = vᵢ - ⟨v⟩ are velocity fluctuations from mean
        θ(·) is the Heaviside step function
        r₀ is the interaction cutoff distance

    Expected ranges (Attanasi et al. 2014):
        Disordered phase : χ/N < 0.3
        Near-criticality : χ/N ∈ [0.3, 1.5]
        Over-correlated  : χ/N > 1.5

    At criticality, χ/N ~ O(1) indicates scale-free correlations where
    perturbations propagate across the entire flock without saturation,
    enabling rapid collective response characteristic of murmurations.
    """

    def evaluate(self, batch: FlockBatch) -> Tensor:
        """
        Compute per-agent susceptibility χ/N.

        Args:
            batch: PyG Batch containing positions [B*N, 3] and velocities [B*N, 3]

        Returns:
            Mean χ/N across batch as scalar tensor
        """
        correlations, = self._reshape_features(batch, 'fluct_corr')
        distances,    = self._reshape_features(batch, 'distances')

        correlation_within_cutoff = th.where(
            distances < self.interaction_cutoff,
            correlations,
            th.zeros_like(distances)
        )

        susceptibility_per_agent = (
            (correlation_within_cutoff.sum(dim=(1, 2)) - self.agent_count) /
            (self.agent_count ** 2)
        )

        return susceptibility_per_agent.mean()


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


class ThermalReactivity(BaseMetric):
    """
    Measure collective thermal response via velocity-temperature correlation.

    Quantifies how agent velocity magnitudes spatially correlate with local
    temperature, capturing the flock's instantaneous responsiveness to thermal
    threats. The Pearson correlation coefficient ρ ranges from -1 to 1:

        ρ(|v|, T) = Cov(|v|, T) / (σ_|v| · σ_T)

    Strong negative correlation (ρ → -1) indicates agents actively flee hot zones,
    demonstrating collective threat response. Weak or positive correlation suggests
    thermal threats are being ignored or approached.

    This instantaneous metric provides a snapshot of threat responsiveness without
    requiring temporal comparisons, making it suitable for shuffled training data.

    Expected ranges:
        - Strong evasion    : ρ ∈ [-1.0, -0.5]
        - Moderate response : ρ ∈ [-0.5, -0.2]
        - Weak response     : ρ ∈ [-0.2,  0.2]
        - Approaching heat  : ρ ∈ [ 0.2,  1.0]
    """

    def evaluate(self, batch: FlockBatch) -> Tensor:
        """
        Compute Pearson correlation between velocity magnitude and temperature.

        Uses cached velocity magnitude for efficient computation. Calculates
        Pearson correlation using the standard formula:

            ρ = Cov(X, Y) / (σ_X · σ_Y)

        Args:
            batch: PyG Batch with velocity [B*N, 3] and temperature [B*N, 1]

        Returns:
            Mean correlation coefficient as scalar tensor
        """
        vel_mag,      = self._reshape_features(batch, 'vel_mag')
        temperature,  = self._reshape_features(batch, 'temperature')
        vel_centered  = vel_mag - vel_mag.mean(dim=1, keepdim=True)
        temp_centered = (
            (temp := temperature.squeeze(-1))
            - temp.mean(dim=1, keepdim=True)
        )

        return (
            (vel_centered * temp_centered).mean(dim=1) /
            (vel_mag.std(dim=1) * temp.std(dim=1) + 1e-8)
        ).mean()


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

        Uses cached velocity magnitude for efficiency when computed alongside
        other metrics like ThermalReactivity.

        Args:
            batch: PyG Batch containing velocity tensor [B*N, 3]

        Returns:
            Mean velocity magnitude as scalar tensor
        """
        vel_mag, = self._reshape_features(batch, 'vel_mag')
        return vel_mag.mean()


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
        murmuration : MurmurationModel
    ):
        """
        Initialize factory with configuration models.

        Args:
            agent_count : Number of agents in the flock
            metrics     : Metrics configuration for thresholds and parameters
            murmuration : Murmuration dynamics configuration
        """
        self.cfg = {
            "agent_count"           : agent_count,
            "correlation_threshold" : metrics.correlation_threshold,
            "coupling_decay"        : murmuration.coupling_decay,
            "heterogeneity_std"     : murmuration.heterogeneity_std,
            "interaction_cutoff"    : metrics.interaction_cutoff,
            "j_base"                : murmuration.j_base,
            "wave_radius"           : metrics.wave_radius,
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
                "clustering_coefficient" : make(ClusteringCoefficient),
                "correlation_length"     : make(CorrelationLength),
                "fiedler_value"          : make(FiedlerValue),
                "mae"                    : make(MAE),
                "max_entropy_energy"     : make(MaxEntropyEnergy),
                "noise_heterogeneity"    : make(NoiseHeterogeneity),
                "orientation_wave"       : make(OrientationWave),
                "pairwise_coherence"     : make(PairwiseCoherence),
                "polarization"           : make(Polarization),
                "r2"                     : make(R2),
                "rmse"                   : make(RMSE),
                "susceptibility"         : make(Susceptibility),
                "temperature"            : make(Temperature),
                "thermal_reactivity"     : make(ThermalReactivity),
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
