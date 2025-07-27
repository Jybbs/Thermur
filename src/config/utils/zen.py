"""
Hydra-zen configuration builders with project-standard defaults.

This module centralizes all hydra-zen functionality to ensure consistency
across configuration domains. It provides enhanced versions of hydra-zen's
core functions that apply Thermur-specific defaults automatically.

The primary exports are:
- store: Direct re-export for creating configuration stores
- build: Enhanced builds function with automatic signature population
- make_config: Direct re-export for creating structured configs

By importing from this module rather than hydra-zen directly, we ensure:
1. Consistent configuration patterns across the project
2. Easier migration if we need to change defaults
3. A single place to extend hydra-zen functionality
"""
from hydra_zen import builds, store
from hydra_zen import make_custom_builds_fn, make_config

build = make_custom_builds_fn(
    populate_full_signature = True,
    zen_partial             = True
)
"""
Standard build function for Thermur configurations.

This custom builds function applies our project-wide defaults:
- populate_full_signature: Automatically includes all parameters from the 
  target's signature in the generated config for better IDE support and 
  runtime validation
- zen_partial: Enables partial instantiation, allowing runtime injection
  of parameters (useful for dependency injection patterns)

Example:
    >>> from config.utils.zen import store, build
    >>> 
    >>> optimizer_store = store(group="optimizer")
    >>> 
    >>> @optimizer_store(name="adam")
    >>> def adam_build():
    >>>     return build(
    >>>         torch.optim.Adam,
    >>>         lr = 1e-3,
    >>>         weight_decay = 1e-4
    >>>     )
"""