"""
Pydantic models for imitation learning configuration validation.

These schemas define the structure and constraints for all imitation learning
components. The models enforce type safety and parameter bounds, ensuring that
configurations are valid before runtime instantiation.

Key model categories:
- Control: Reynolds weights ω_coh, ω_sep, ω_align and numerical stability parameters
- Learning: Neural network architecture, optimization, and training hyperparameters
- Safety: Thermal barriers, collision avoidance, and control bounds
- Monitoring: Logging, metrics, and experiment tracking with Weights & Biases
- Visualization: Real-time rendering of flock dynamics and thermal fields
"""
from .control       import *
from .learning      import *
from .monitoring    import *
from .physics       import *
from .safety        import *
from .specs         import *
from .flock         import *
from .visualization import *