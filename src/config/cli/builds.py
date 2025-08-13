"""
CLI configuration container.

Provides a unified configuration object for all CLI components using Pydantic
BaseModel.

All configurations are pre-instantiated for immediate use with minimal overhead.
"""
from .schemas import *
from pydantic import BaseModel


class CLIConfiguration(BaseModel):
    """
    Configuration container for CLI components.

    Uses Pydantic BaseModel with pre-instantiated config objects to provide
    type safety and validation while maintaining fast CLI startup times.
    """
    display  : DisplayModel  = DisplayModel()
    download : DownloadModel = DownloadModel()
    secrets  : GlobusSecrets = GlobusSecrets()
    wandb    : WandbModel    = WandbModel()

cfg = CLIConfiguration()
