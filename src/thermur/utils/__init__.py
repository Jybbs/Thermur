"""
General utilities for system configuration and data management.

This package provides cross-cutting functionality used throughout Thermur:
- Logging configuration with structured output via Loguru
- Random seed management for reproducible experiments
- XML generation for dynamic MuJoCo model creation
- Data source abstraction for flexible environment interfaces

These utilities follow dependency injection patterns to maintain modularity
and testability across the codebase.
"""
from .data    import EnvironmentDataSource
from .logging import configure_loguru
from .seed    import set_seed
from .xml     import generate_swarm_xml, load_swarm_model