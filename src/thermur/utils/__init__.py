"""
General utilities for system configuration and data management.

This package provides cross-cutting functionality used throughout Thermur:
- Logging configuration with structured output via Loguru
- Random seed management for reproducible experiments
- XML generation for dynamic MuJoCo model creation

These utilities follow dependency injection patterns to maintain modularity
and testability across the codebase.
"""
from .logging import configure_loguru
from .seed    import set_seed
from .xml     import generate_flock_xml, load_flock_model