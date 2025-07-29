"""
CLI configuration system.

This module provides pure data configurations for the CLI framework. Since these
are static settings that don't require runtime instantiation or interpolation,
they are created as simple Pydantic model instances.

The configuration is organized by domain:
- display  : Terminal UI themes and formatting  
- download : Globus data transfer settings
- secrets  : Secure token storage
- wandb    : Weights & Biases integration settings

All configurations are defined in schemas.py with full documentation and
instantiated in stores.py for direct use by CLI components.
"""
from .schemas import *
