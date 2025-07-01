"""
Hydra-zen builders for imitation learning components.

These factories create instantiatable configurations for the core imitation learning
pipeline. Each builder wraps component constructors with Hydra-zen metadata while
preserving Pydantic validation through the zen() wrapper.

The builders are organized by functionality:
- Data: Trajectory collection and experience replay
- Control: Expert flocking controller and safety mechanisms
- Learning: GNN policy, optimizer, and loss functions
- Simulation: MuJoCo environment and flock dynamics
- Visualization: Real-time rendering and debugging tools
"""
from .data          import *
from .controller    import *
from .imitation     import *
from .monitoring    import *
from .safety        import *
from .simulation    import *
from .flock         import *
from .visualization import *
