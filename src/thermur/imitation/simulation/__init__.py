"""
Simulation environment for thermally-aware drone flocks.

This package provides the core simulation components for training and evaluating
multi-agent flocking policies in wildfire scenarios. The main components include:

- SimulationEnv: TorchRL-compatible environment managing agent physics and observations
- WRFDataSource: Loader for environmental data (wind, temperature fields)
- XMLGenerator: Dynamic MuJoCo model generation for N-agent systems

The simulation integrates rigid-body physics (MuJoCo) with dynamic environmental
hazards, providing a realistic testbed for thermal-aware navigation policies.
"""