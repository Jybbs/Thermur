"""
Hydra-zen configuration factories for CLI components.

This module provides builders that create Hydra-compatible configurations
for the CLI system. It follows the same clean patterns as other factories
in the project.
"""
from config.cli.schemas.application import CLIModel
from config.cli.schemas.display     import DisplayModel, MessagesModel
from config.cli.schemas.interaction import DownloadModel, PresetsModel, PromptsModel
from config.imitation.schemas.wandb import WandbModel
from hydra_zen                      import builds

build_cli      = builds(CLIModel)
build_display  = builds(DisplayModel)
build_download = builds(DownloadModel)
build_messages = builds(MessagesModel)
build_presets  = builds(PresetsModel)
build_prompts  = builds(PromptsModel)
build_wandb    = builds(WandbModel)