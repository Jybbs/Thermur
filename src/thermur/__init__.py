"""
Thermur: Thermally-aware drone flock control via imitation learning.

This package implements a complete system for training autonomous drone flocks
to navigate wildfire scenarios using graph neural networks and behavioral cloning.
The architecture consists of several key components:

1. Control: Expert controllers implementing Reynolds flocking rules with thermal
   constraints, providing demonstrations for imitation learning.

2. Models: Graph Neural Network policies that process flock states as dynamic
   graphs and output control actions.

3. Simulation: MuJoCo-based physics environment modeling drone dynamics and
   thermal field interactions.

4. Training: Imitation learning algorithms for training policies from expert
   demonstrations using behavioral cloning.

5. Visualization: Real-time 3D rendering of flock dynamics and thermal fields
   for debugging and analysis.

The system is designed for modularity and extensibility, with clean interfaces
between components and comprehensive configuration management through Hydra-zen.
"""