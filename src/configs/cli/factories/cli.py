"""
Hydra-zen configuration factories for CLI components.

This module provides builders that create Hydra-compatible configurations
for the CLI system. It follows the same clean patterns as other factories
in the project.
"""
from hydra_zen import builds
from ..schemas.core import CLIModel, CommandsModel, TrainingComponentsModel
from ..schemas.explorer import ExplorerMessagesModel, ExplorerModel
from ..schemas.messages import (
    HeadersModel,
    MessagesModel,
    MessageTypesModel,
    PromptsModel,
    SectionsModel,
    StatusModel,
)
from ..schemas.presets import PresetsModel, TipsModel
from ..schemas.system import SystemModel, ValidationModel, WandbDisplayModel
from ..schemas.ui import ThemeModel, UIModel

build_cli                 = builds(CLIModel)
build_commands            = builds(CommandsModel)  
build_explorer            = builds(ExplorerModel)
build_explorer_messages   = builds(ExplorerMessagesModel)
build_headers             = builds(HeadersModel)
build_messages            = builds(MessagesModel)
build_message_types       = builds(MessageTypesModel)
build_presets             = builds(PresetsModel)
build_prompts             = builds(PromptsModel)
build_sections            = builds(SectionsModel)
build_status              = builds(StatusModel)
build_system              = builds(SystemModel)
build_theme               = builds(ThemeModel)
build_tips                = builds(TipsModel)
build_training_components = builds(TrainingComponentsModel)
build_ui                  = builds(UIModel)
build_validation          = builds(ValidationModel)
build_wandb_display       = builds(WandbDisplayModel)