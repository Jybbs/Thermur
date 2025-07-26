"""Controller configuration schemas.

This subpackage contains configuration models for expert control algorithms
used in demonstration generation:

- flock.py: Reynolds flocking algorithm configuration with thermal constraints
- safety.py: Control Barrier Function (CBF) safety filter configuration

The controller configurations define parameters for generating expert
demonstrations that balance collective behavior objectives (cohesion,
alignment, separation) with safety constraints (collision avoidance,
thermal limits). These demonstrations serve as training data for the
imitation learning policy.
"""
from .flock  import *
from .safety import *