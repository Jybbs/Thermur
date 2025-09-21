"""
Adaptive Lanczos-based Fiedler value implementation for PyTorch Lightning.

Computes the second smallest eigenvalue (Fiedler value) of graph Laplacians
using iterative Lanczos algorithm with automatic convergence detection.
"""
from __future__            import annotations
from torch_geometric.data  import Data, Batch
from torch_geometric.utils import to_dense_adj
from typing                import TYPE_CHECKING

if TYPE_CHECKING:
    from config.types import FlockBatch
    from torch        import Tensor

import torch as th


class LanczosFiedlerValue:
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

    def __init__(self, agent_count: int):
        """
        Initialize Lanczos Fiedler value computation.

        Args:
            agent_count: Number of agents per graph
        """
        self.agent_count = agent_count
        self.k_used      = None

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
                edge_index    = batch.edge_index,
                batch         = batch.batch,
                max_num_nodes = self.agent_count
            )
        ) + adj.mT

        A.sign_()
        return th.diag_embed(A.sum(dim=-1)) - A
    
    def _build_tridiagonal(self, alphas: Tensor, betas: Tensor) -> Tensor:
        """
        Construct symmetric tridiagonal matrices from Lanczos coefficients.

        Assembles the k×k tridiagonal matrix T that approximates the spectral
        properties of the original N×N Laplacian. Building on CPU avoids
        repeated device transfers during eigenvalue computation.

        Args:
            alphas: Diagonal coefficients from Lanczos iteration [B, k]
            betas: Off-diagonal coefficients (coupling terms) [B, k-1]

        Returns:
            Symmetric tridiagonal matrices [B, k, k]
        """
        _, k      = alphas.shape
        diagonals = th.diag_embed(alphas.cpu())

        if betas.shape[1] > 0:
            off_diagonals = betas.cpu()[:, :k - 1]
            diagonals[:, range(k - 1),  range(1, k)]  = off_diagonals
            diagonals[:, range(1, k),   range(k - 1)] = off_diagonals

        return diagonals

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

        initial_vec     = th.randn(B, N, device=device)
        initial_vec    -= initial_vec.mean(dim=1, keepdim=True)
        v_curr = v_prev = initial_vec / initial_vec.norm(dim=1, keepdim=True)
        prev_fiedler    = th.zeros(B)

        for i in range(self.agent_count):
            w = th.einsum('bij, bj -> bi', laplacians, v_curr)

            alpha = th.einsum('bi, bi -> b', w, v_curr)
            alphas[:, i] = alpha

            w = w - alpha[:, None] * v_curr - (
                betas[:, i-1, None] * v_prev if i > 0 else 0
            )
            w = w - w.mean(dim=1, keepdim=True)

            if (beta := w.norm(dim=1)).min() <= 1e-10 or i >= self.agent_count - 1:
                break

            betas[:, i] = beta
            v_prev = v_curr
            v_curr = w / beta[:, None]
            if i >= 10 and i % 5 == 0:
                if th.allclose(
                    curr_fiedler := th.linalg.eigvalsh(
                        self._build_tridiagonal(
                            alphas[:, :(k := i + 1)],
                            betas[:, :k-1]
                        )
                    )[:, 0],
                    prev_fiedler
                ):
                    self.k_used = k
                    return curr_fiedler.to(device)

                prev_fiedler = curr_fiedler

        self.k_used = i + 1
        T_final = self._build_tridiagonal(
            alphas = alphas[:, :self.k_used],
            betas  = betas[:, :self.k_used-1]
        )
        return th.linalg.eigvalsh(T_final)[:, 0].to(device)

    def compute(self, batch: FlockBatch) -> Tensor:
        """
        Compute harmonic mean of Fiedler values.

        The harmonic mean H = n/Σ(1/λᵢ) emphasizes weak connectivity,
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

############## TESTING SECTION
##############################

def create_test_batch(batch_size: int, agent_count: int = 50, k_neighbors: int = 7) -> Batch:
    """
    Create a test batch similar to actual training data.
    """
    device = th.device('mps')

    data_list = []
    for _ in range(batch_size):
        # Random positions for agents
        position = th.randn(agent_count, 3, device=device) * 10

        # k-NN connectivity
        distances    = th.cdist(position, position)
        _, neighbors = distances.topk(k_neighbors + 1, largest=False)
        neighbors    = neighbors[:, 1:]  # Remove self

        # Build edge list
        src = th.arange(agent_count, device=device).repeat_interleave(k_neighbors)
        dst = neighbors.flatten()
        edge_index = th.stack([src, dst])

        data = Data(
            action       = th.randn(agent_count, 3, device=device),
            alert_states = th.zeros(agent_count, 1, device=device),
            edge_index   = edge_index,
            position     = position,
            temperature  = th.rand(agent_count, 1, device=device),
            velocity     = th.randn(agent_count, 3, device=device),
            num_nodes    = agent_count,  # Explicitly set to avoid warning
        )
        data_list.append(data)

    # Create batch - PyG automatically creates batch tensor on CPU
    batch = Batch.from_data_list(data_list)

    # Move batch tensor to same device as data
    if hasattr(batch, 'batch') and batch.batch is not None:
        batch.batch = batch.batch.to(device)

    return batch



if __name__ == "__main__":
    import time
    import numpy as np

    print("=" * 60)
    print("Lanczos Fiedler Value Benchmark")
    print("=" * 60)

    batch_sizes = [32, 64, 128]
    agent_count = 50
    k_neighbors = 7
    n_warmup    = 10
    n_runs      = 30

    print(f"\nConfiguration: {agent_count} agents, {k_neighbors} neighbors")
    print(f"Warmup: {n_warmup} runs, Benchmark: {n_runs} runs")
    print("-" * 60)

    for batch_size in batch_sizes:
        batch   = create_test_batch(batch_size, agent_count, k_neighbors)
        lanczos = LanczosFiedlerValue()

        # Warmup
        for _ in range(n_warmup):
            _ = lanczos.compute(batch)
            th.mps.synchronize()

        # Benchmark
        times = []
        for _ in range(n_runs):
            th.mps.synchronize()
            start  = time.perf_counter()
            result = lanczos.compute(batch)
            th.mps.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        mean_time = np.mean(times)
        std_time  = np.std(times)
        min_time  = np.min(times)
        max_time  = np.max(times)
        k_used    = lanczos.k_used if lanczos.k_used else 0

        print(f"\nBatch {batch_size:3d}:")
        print(f"  Mean: {mean_time:6.2f} ± {std_time:4.2f} ms ({mean_time/batch_size:5.3f} ms/graph)")
        print(f"  Range: [{min_time:5.2f}, {max_time:5.2f}] ms")
        print(f"  k={k_used} ({100*k_used/agent_count:.0f}% of graph), value={result.item():.4f}")

    print("=" * 60)