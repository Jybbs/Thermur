"""Simulation configuration schemas.

This subpackage contains configuration models for the physics simulation
and environment management:

- environment.py: MuJoCo physics simulation and TorchRL environment configuration
- loader.py: Weather Research and Forecasting (WRF) data loading configuration

The simulation configurations define the physical world model, including
drone dynamics, environmental forces (wind, thermals), and data sources.
These components create a realistic training environment that captures
the challenges of coordinated flight in dynamic atmospheric conditions.
"""
from .environment import *
from .loader      import *