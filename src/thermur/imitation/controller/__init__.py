"""
Control algorithms for thermally-aware drone flocks.

This package implements murmuration dynamics based on starling flocks that
generates demonstrations for imitation learning. The core algorithm uses
topological interactions (k-nearest neighbors) rather than metric distances,
maintaining critical state dynamics for rapid information propagation.

The murmuration controller implements an enhanced Hamiltonian formulation:

    E = -Σ J_ij 𝐬_i · 𝐬_j - Σ 𝐡_i · 𝐬_i

where 𝐬_i are normalized velocities and J_ij decay with topological distance.

The flock exhibits two distinct modes:
- Cruise : Standard Reynolds rules with susceptibility-modulated alignment
- Alert  : Enhanced correlation and density for rapid threat response

The SafetyFilter ensures all control actions respect physical constraints and
maintain safe distances from thermal hazards using Control Barrier Functions.
"""
from .murmuration import *
from .safety      import *
