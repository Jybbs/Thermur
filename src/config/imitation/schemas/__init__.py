"""
Schema definitions for imitation learning configuration.

This package contains Pydantic models that define the configuration structure
for the imitation learning system. The schemas are organized to mirror the
structure of the thermur.imitation module:

- controller/: Configuration for expert control algorithms (flocking, safety)
- lightning/: PyTorch Lightning training infrastructure configuration
- monitoring/: Metrics collection and event logging configuration
- simulation/: Physics simulation and environment configuration
- visualization/: 3D rendering and visual monitoring configuration

The schemas use Pydantic for validation and type safety, ensuring that
configuration values meet the requirements of the underlying implementation.
Default values are provided where appropriate to minimize configuration
complexity for common use cases.
"""
