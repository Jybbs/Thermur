"""
Thermur configuration system built on Hydra-zen and Pydantic.

This package provides a domain-based organization for all configurations, ensuring
type safety through Pydantic validation and runtime flexibility through Hydra-zen's
instantiation system. The configuration hierarchy is organized into two main domains:

- CLI: User interface configuration for the command-line tool
- Imitation: Machine learning configuration for behavioral cloning

Each domain contains schemas (Pydantic models), factories (Hydra-zen builders), and
workloads (composed configurations). This structure enables clean separation of concerns
while maintaining a unified configuration interface.

The system supports both static validation at configuration time and dynamic instantiation
at runtime, providing the best of both worlds for complex ML systems.
"""