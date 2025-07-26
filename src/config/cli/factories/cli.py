"""
Hydra-zen configuration factories for CLI components.

This module provides builders that create Hydra-compatible configurations
for the CLI system. It follows the same clean patterns as other factories
in the project.
"""
from config.cli.schemas                 import *
from config.imitation.schemas.lightning import WandbModel
from hydra_zen                          import builds

build_display  = builds(DisplayModel)
build_download = builds(DownloadModel)
build_messages = builds(MessagesModel)
build_prompts  = builds(PromptsModel)
build_secrets  = builds(GlobusSecrets)
build_system   = builds(SystemModel)
build_wandb    = builds(WandbModel)