"""
Hydra-zen configuration builders with project-standard defaults.

This module centralizes all hydra-zen functionality to ensure consistency
across configuration domains. It provides enhanced versions of hydra-zen's
core functions that apply Thermur-specific defaults automatically.

The primary exports are:
- store: Direct re-export from hydra-zen for creating config stores
- thermur_build: Enhanced builds function with automatic signature population
- thermur_make_all: Helper to create "all" configs and register with Hydra
- make_config: Direct re-export for creating structured configs

By importing from this module rather than hydra-zen directly, we ensure:
1. Consistent configuration patterns across the project
2. Easier migration if we need to change defaults
3. A single place to extend hydra-zen functionality
"""
from hydra_zen import store, make_custom_builds_fn, make_config

thermur_build = make_custom_builds_fn(
    populate_full_signature = True,
    zen_partial             = True
)
"""
Thermur-specific build function for configurations.

This custom builds function applies our project-wide defaults:
- populate_full_signature: Automatically includes all parameters from the 
  target's signature in the generated config for better IDE support and 
  runtime validation
- zen_partial: Enables partial instantiation, allowing runtime injection
  of parameters (useful for dependency injection patterns)

Example:
    >>> from config.utils.zen import thermur_store, thermur_build
    >>> 
    >>> optimizer_store = thermur_store(group="optimizer")
    >>> 
    >>> @optimizer_store(name="adam")
    >>> def adam_build():
    >>>     return thermur_build(
    >>>         torch.optim.Adam,
    >>>         lr = 1e-3,
    >>>         weight_decay = 1e-4
    >>>     )
"""

def thermur_make_all(store_instance):
    """
    Create an "all" config that contains all other configs in the store as fields,
    then register everything with Hydra's ConfigStore.
    
    This should be called at the end of each stores module after all configs
    have been registered with the store.
    
    Example:
        controller = store()(group="controller")
        
        @controller(name="expert")
        def expert_build(): ...
        
        thermur_make_all(controller)  # Creates controller.all and registers with Hydra
    """
    if not store_instance.groups:
        raise ValueError("Store must have a group")
    
    group = store_instance.groups[0]
    names = [name for (g, name) in store_instance._internal_repo.keys() 
             if g == group and name != "all"]

    all_config = make_config(**{
        name: f"${{{group}.{name}}}" 
        for name in sorted(names)
    })
    
    store_instance(all_config, name="all")
    store_instance.add_to_hydra_store()