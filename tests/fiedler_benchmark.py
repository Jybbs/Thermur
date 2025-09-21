"""
Adaptive Lanczos-based FiedlerValue implementation for PyTorch Lightning.
Automatically determines optimal iterations by monitoring convergence.
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
    Production-ready Fiedler value computation using adaptive Lanczos.

    Automatically determines optimal iteration count by monitoring convergence
    of the Fiedler value (second smallest eigenvalue). No manual tuning required.

    Key features:
    - Converges in 10-70 iterations depending on graph size and structure
    - Monitors relative change in Fiedler value every 5 iterations
    - Minimal overhead from convergence checking
    - Works efficiently from 10 to 1000+ agent graphs
    """

    def __init__(
        self,
        agent_count    : int   = 50,
        check_interval : int   = 5,
        min_iterations : int   = 10
    ):
        """
        Args:
            agent_count    : Number of agents per graph
            check_interval : How often to check convergence (default 5)
            min_iterations : Minimum Lanczos iterations before checking convergence
        """
        self.agent_count    = agent_count
        self.check_interval = check_interval
        self.min_iterations = min_iterations
        self.k_used         = None  # Track actual iterations used

    def _build_laplacian_batch(self, batch: FlockBatch) -> Tensor:
        # Build adjacency matrix
        A = to_dense_adj(
            edge_index    = batch.edge_index,
            batch         = batch.batch,
            max_num_nodes = self.agent_count
        )

        # Optimized Laplacian construction
        A = (A + A.mT).sign_()  # Symmetrize and binarize
        D = A.sum(dim=-1)
        return th.diag_embed(D) - A

    def _lanczos_eigenvalue(self, L: Tensor) -> Tensor:
        B, N, _ = L.shape
        device  = L.device

        # Pre-allocate storage
        alpha_storage = th.zeros(B, self.agent_count,     device=device)
        beta_storage  = th.zeros(B, self.agent_count - 1, device=device)

        # Initialize vectors as (B, N) - simpler broadcasting
        v_init  = th.randn(B, N, device=device)
        v_init -= v_init.mean(dim=1, keepdim=True)
        v_curr  = v_prev = v_init / v_init.norm(dim=1, keepdim=True)

        prev_fiedler = th.zeros(B)  # CPU tensor for convergence checking

        for i in range(self.agent_count):
            # Matrix-vector product using einsum - no reshape needed
            w = th.einsum('bij, bj -> bi', L, v_curr)

            # Compute alpha (dot product per batch)
            alpha = th.einsum('bi, bi -> b', w, v_curr)
            alpha_storage[:, i] = alpha

            # Orthogonalize using broadcasting (automatic with einsum output shape)
            w = w - alpha[:, None] * v_curr
            if i > 0:
                w = w - beta_storage[:, i-1, None] * v_prev

            # Remove nullspace component
            w = w - w.mean(dim=1, keepdim=True)

            # Compute beta
            beta = w.norm(dim=1)

            # Check for breakdown
            if beta.min() <= 1e-10 or i >= self.agent_count - 1:
                break

            beta_storage[:, i] = beta

            # Update vectors for next iteration
            v_prev = v_curr
            v_curr = w / beta[:, None]

            # Check convergence periodically after minimum iterations
            if i >= self.min_iterations and i % self.check_interval == 0:
                k = i + 1
                T_small = self._build_tridiagonal(
                    alpha = alpha_storage[:, :k],
                    beta  = beta_storage[:, :k-1]
                )
                # Already on CPU from build_tridiagonal
                curr_fiedler = th.linalg.eigvalsh(T_small)[:, 0]

                # Check convergence using relative tolerance (on CPU)
                if th.allclose(curr_fiedler, prev_fiedler, atol=0):
                    self.k_used = k
                    return curr_fiedler.to(device)

                prev_fiedler = curr_fiedler

        # Final computation
        self.k_used = i + 1
        T_final = self._build_tridiagonal(
            alpha = alpha_storage[:, :self.k_used],
            beta  = beta_storage[:, :self.k_used-1]
        )
        # Already on CPU from build_tridiagonal
        return th.linalg.eigvalsh(T_final)[:, 0].to(device)


    def _build_tridiagonal(self, alpha: Tensor, beta: Tensor) -> Tensor:
        _, k = alpha.shape

        # Build on CPU to avoid repeated large matrix transfers
        alpha_cpu = alpha.cpu()
        T = th.diag_embed(alpha_cpu)

        if beta.shape[1] > 0:
            # Add upper and lower diagonals
            beta_cpu = beta.cpu()[:, :k-1]
            T[:, range(k-1), range(1, k)] = beta_cpu
            T[:, range(1, k), range(k-1)] = beta_cpu

        return T


    def compute(self, batch: FlockBatch) -> Tensor:
        laplacians = self._build_laplacian_batch(batch)
        fiedler    = self._lanczos_eigenvalue(laplacians).clamp_min(1e-10)
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