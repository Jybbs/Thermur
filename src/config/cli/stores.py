"""
CLI configuration stores.

Provides a clean namespace for all CLI configurations using SimpleNamespace.
All configs are instantiated Pydantic models for direct use by CLI components.
"""
from .schemas import *
from types    import SimpleNamespace

cfg = SimpleNamespace(
    display  = DisplayModel(),
    download = DownloadModel(),
    secrets  = GlobusSecrets(),
    wandb    = WandbConfig()
)