"""
Adaptive Lanczos-based FiedlerValue implementation for PyTorch Lightning.
Automatically determines optimal iterations by monitoring convergence.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj
import time
import numpy as np


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
        max_iterations : int   = 150, 
        min_iterations : int   = 10,
        tolerance      : float = 1e-4,
    ):
        """
        Args:
            agent_count    : Number of agents per graph
            check_interval : How often to check convergence (default 5)
            max_iterations : Maximum iterations (safety cap)
            min_iterations : Minimum Lanczos iterations before checking convergence
            tolerance      : Relative convergence tolerance for Fiedler value
        """
        self.agent_count    = agent_count
        self.min_k          = min_iterations
        self.max_k          = min(max_iterations, agent_count - 1)
        self.tolerance      = tolerance
        self.check_interval = check_interval
        self.k_used         = None  # Track actual iterations used

        # No additional initialization needed

    def _build_laplacians_vectorized(self, batch: Batch) -> torch.Tensor:
        device = batch.edge_index.device

        # Ensure batch assignment exists and is on correct device
        if not hasattr(batch, 'batch'):
            batch.batch = torch.zeros(batch.num_nodes, dtype=torch.long, device=device)
        elif batch.batch.device != device:
            batch.batch = batch.batch.to(device)

        # Single-pass adjacency matrix construction
        A = to_dense_adj(
            edge_index=batch.edge_index,
            batch=batch.batch,
            max_num_nodes=self.agent_count
        )

        # Fused symmetrization and Laplacian computation
        # Use in-place operations where possible to reduce memory allocation
        A = A + A.mT  # Symmetrize
        A = A.bool().float()  # Binary adjacency
        D = A.sum(dim=-1, keepdim=True)  # Degree vector
        L = torch.zeros_like(A)
        L.diagonal(dim1=-2, dim2=-1).copy_(D.squeeze(-1))  # Set diagonal
        return L - A  # Return Laplacian

    def _lanczos_iteration(self, L: torch.Tensor) -> torch.Tensor:
        B, N, _ = L.shape
        device = L.device

        # Pre-allocate storage for better memory efficiency
        alpha_storage = torch.zeros(B, self.max_k,     device=device)
        beta_storage  = torch.zeros(B, self.max_k - 1, device=device)

        # Initialize with random vector orthogonal to nullspace
        v_init = torch.randn(B, N, device=device)
        v_prev = F.normalize(v_init - v_init.mean(dim=1, keepdim=True), p=2, dim=1)
        v_curr = v_prev

        # Track convergence
        prev_fiedler = None
        actual_k = 0

        for i in range(self.max_k):
            # Matrix-vector product
            w = torch.bmm(L, v_curr.unsqueeze(-1)).squeeze(-1)

            # Compute alpha (diagonal element)
            alpha = (w * v_curr).sum(dim=-1)
            alpha_storage[:, i] = alpha

            # Orthogonalize
            w = w - alpha.unsqueeze(1) * v_curr
            if i > 0:
                w = w - beta_storage[:, i-1].unsqueeze(1) * v_prev

            # Remove nullspace component
            w = w - w.mean(dim=1, keepdim=True)

            # Compute beta and normalize
            beta = w.norm(dim=1)

            # Check for breakdown
            if beta.min() <= 1e-10:
                actual_k = i + 1
                break

            beta_storage[:, i] = beta

            # Update vectors for next iteration
            v_prev = v_curr
            v_curr = w / beta.unsqueeze(1).clamp_min(1e-10)

            # Check convergence periodically (reduced overhead version)
            if i >= self.min_k and (i - self.min_k) % self.check_interval == 0:
                # Build small tridiagonal matrix for convergence check
                k_check = i + 1
                alpha_check = alpha_storage[:, :k_check]
                beta_check = beta_storage[:, :i] if i > 0 else None
                T_small = self._build_tridiagonal(alpha_check, beta_check)

                # Get smallest eigenvalue
                curr_fiedler = torch.linalg.eigvalsh(T_small)[:, 0].to(device)

                if self._has_converged(curr_fiedler, prev_fiedler):
                    self.k_used = i + 1
                    return curr_fiedler

                prev_fiedler = curr_fiedler

            actual_k = i + 1

        # Final computation with actual iterations used
        self.k_used = actual_k
        alpha_final = alpha_storage[:, :actual_k]
        beta_final = beta_storage[:, :actual_k-1] if actual_k > 1 else None
        return self._compute_fiedler_fast(alpha_final, beta_final, device)


    def _compute_fiedler_fast(
        self, 
        alpha  : torch.Tensor, 
        beta   : torch.Tensor | None, 
        device
    ) -> torch.Tensor:
        B, k = alpha.shape
        if k == 0:
            return torch.zeros(B, device=device)

        # Build and diagonalize tridiagonal matrix
        T = self._build_tridiagonal(alpha, beta)
        eigvals = torch.linalg.eigvalsh(T)

        # Return smallest eigenvalue to original device
        return eigvals[:, 0].to(device)

    def _has_converged(self, curr_fiedler: torch.Tensor,
                       prev_fiedler: torch.Tensor | None) -> bool:
        if prev_fiedler is None:
            return False

        rel_change = (curr_fiedler - prev_fiedler).abs() / (curr_fiedler.abs() + 1e-10)
        return rel_change.max().item() < self.tolerance

    def _build_tridiagonal(self, alpha: torch.Tensor, beta: torch.Tensor | None) -> torch.Tensor:
        B, k = alpha.shape

        # Always work on CPU for eigendecomposition on MPS
        alpha = alpha.cpu()
        beta  = beta.cpu() if beta is not None else None

        # Initialize tridiagonal matrix
        T = torch.zeros(B, k, k, device='cpu')

        # Set diagonal using advanced indexing (vectorized)
        batch_idx = torch.arange(B)[:, None]
        diag_idx  = torch.arange(k)
        T[batch_idx, diag_idx, diag_idx] = alpha

        # Set off-diagonals if present
        if beta is not None and k > 1:
            k_beta  = beta.shape[1]
            off_idx = torch.arange(min(k_beta, k-1))
            T[batch_idx, off_idx, off_idx + 1] = beta[:, :min(k_beta, k-1)]
            T[batch_idx, off_idx + 1, off_idx] = beta[:, :min(k_beta, k-1)]

        return T


    def compute(self, batch: Batch) -> torch.Tensor:
        """Compute Fiedler value with adaptive convergence monitoring."""
        laplacians = self._build_laplacians_vectorized(batch)
        fiedler    = self._lanczos_iteration(laplacians).clamp_min(1e-10)
        return fiedler.numel() / fiedler.reciprocal().sum()

############## TESTING SECTION
##############################

def create_test_batch(batch_size: int, agent_count: int = 50, k_neighbors: int = 7):
    """
    Create a test batch similar to actual training data.
    """
    device = torch.device('mps')

    data_list = []
    for _ in range(batch_size):
        # Random positions for agents
        position = torch.randn(agent_count, 3, device=device) * 10

        # k-NN connectivity
        distances    = torch.cdist(position, position)
        _, neighbors = distances.topk(k_neighbors + 1, largest=False)
        neighbors    = neighbors[:, 1:]  # Remove self

        # Build edge list
        src = torch.arange(agent_count, device=device).repeat_interleave(k_neighbors)
        dst = neighbors.flatten()
        edge_index = torch.stack([src, dst])

        data = Data(
            action       = torch.randn(agent_count, 3, device=device),
            alert_states = torch.zeros(agent_count, 1, device=device),
            edge_index   = edge_index,
            position     = position,
            temperature  = torch.rand(agent_count, 1, device=device),
            velocity     = torch.randn(agent_count, 3, device=device),
        )
        data_list.append(data)

    return Batch.from_data_list(data_list)



if __name__ == "__main__":
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
            torch.mps.synchronize()

        # Benchmark
        times = []
        for _ in range(n_runs):
            torch.mps.synchronize()
            start  = time.perf_counter()
            result = lanczos.compute(batch)
            torch.mps.synchronize()
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