"""
Control algorithms for thermally-aware drone flocks and offline demonstrations.

This package implements murmuration dynamics based on starling flocks that
generates demonstrations for imitation learning. The core algorithm uses
topological interactions (k-nearest neighbors) rather than metric distances,
maintaining critical state dynamics for rapid information propagation.

The murmuration controller implements an enhanced Hamiltonian formulation:

    E = -Σ J_ij 𝐬_i · 𝐬_j - Σ 𝐡_i · 𝐬_i

where 𝐬_i are normalized velocities and J_ij decay with topological distance.

The flock exhibits two distinct modes:
- Cruise : Standard murmuration dynamics with susceptibility-modulated alignment
- Alert  : Enhanced correlation and density for rapid threat response

Key components:
- DemonstrationsDataset : PyG InMemoryDataset for offline expert trajectories
- MurmurationController : Expert controller implementing murmuration dynamics
- ThermalPenalty        : Thermal safety constraints using KS penalties

The demonstrations dataset generates and caches expert trajectories for
behavioral cloning, providing a PyTorch Lightning datamodule interface for
efficient batched training.
"""
from .demonstrations import *
from .murmuration    import *
from .safety         import *
