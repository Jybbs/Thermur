"""
Thermur: Thermally-aware drone flock control via imitation learning.

This package implements a complete system for training autonomous drone flocks
to navigate wildfire scenarios using graph neural networks and behavioral cloning.
The architecture consists of several key components:

1. CLI: Command-line interface for training, validation, and monitoring using
   Typer and Rich for an enhanced user experience.

2. Control: Expert controllers implementing Reynolds flocking rules with thermal
   constraints, providing demonstrations for imitation learning.

3. Data: WRF-Fire data loading and interpolation for environmental observations
   including temperature, wind, and fire heat flux fields.

4. Models: Graph Neural Network policies that process flock states as dynamic
   graphs and output control actions.

5. Simulation: MuJoCo-based physics environment modeling drone dynamics and
   thermal field interactions.

6. Training: Imitation learning algorithms for training policies from expert
   demonstrations using behavioral cloning.

7. Visualization: Real-time 3D rendering of flock dynamics and thermal fields
   for debugging and analysis.

The system is designed for modularity and extensibility, with clean interfaces
between components and comprehensive configuration management through Hydra-zen.
"""
