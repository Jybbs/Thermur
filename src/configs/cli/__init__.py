"""
Command-line interface configuration domain.

This package organizes all CLI-related configurations using a layered architecture:
schemas define the data models, factories create Hydra-zen builders, and workloads
compose everything into the final configuration. The CLI configuration controls the
user experience, including themes, prompts, messages, and command definitions.

The configuration system leverages Pydantic for validation and Hydra-zen for 
instantiation, ensuring type safety and runtime flexibility.
"""
from .schemas   import *
from .workloads import cli_cfg, register_cli_cfgs