"""
Simulation environment for thermally-aware drone flocks.

This package provides the core simulation components for training and evaluating
multi-agent flocking policies in wildfire scenarios. The main components include:

- SimulationEnv: TorchRL-compatible environment with simple physics integration
- WRFDataSource: Loader for environmental data (wind, temperature fields)

The simulation uses Euler integration for agent dynamics and interfaces with
environmental data to provide realistic thermal navigation scenarios.
"""
from .environment import *
from .loader      import *
