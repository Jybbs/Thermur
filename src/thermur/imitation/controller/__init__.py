"""
Control algorithms for thermally-aware drone flocks.

This package implements the expert controller that generates demonstrations for
imitation learning. The core algorithm combines Reynolds flocking rules with
thermal constraints through a potential-based approach:

    𝐮_nom = -∇_x U(S_t)

where the potential U incorporates:
- Cohesion: Attraction to local center of mass
- Separation: Repulsion from nearby agents  
- Alignment: Velocity matching with neighbors
- Thermal avoidance: Repulsion from high-temperature regions

The SafetyFilter ensures all control actions respect physical constraints and
maintain safe distances from thermal hazards using Control Barrier Functions.
"""
from .expert import *
from .safety import *