"""
Real-time visualization for flock dynamics and thermal fields.

The Visualizer class provides interactive 3D rendering of the simulation state
using PyVista. Visualization features include:

- Agent trajectories with velocity vectors
- Temperature field as a volumetric heatmap
- Communication graph edges showing network topology
- Safety regions and thermal barriers
- Configurable camera views and rendering styles

The visualization system supports both real-time display during training and
offline rendering of saved trajectories for analysis.
"""
from .visualizer import Visualizer