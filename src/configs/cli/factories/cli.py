"""
Hydra-zen configuration factories for CLI components.

This module provides builders that create Hydra-compatible configurations
for the CLI system. It follows the same clean patterns as other factories
in the project.
"""
from hydra_zen import zen

from ..schemas import (
    CLIModel,
    CommandsModel,
    ExplorerMessagesModel,
    ExplorerModel,
    HeadersModel,
    MessagesModel,
    MessageTypesModel,
    PresetsModel,
    PromptsModel,
    SectionsModel,
    StatusModel,
    SystemModel,
    ThemeModel,
    TipsModel,
    TrainingComponentsModel,
    UIModel,
    ValidationModel,
    WandbDisplayModel,
)

build_cli                 = zen(CLIModel)
build_commands            = zen(CommandsModel)  
build_explorer            = zen(ExplorerModel)
build_explorer_messages   = zen(ExplorerMessagesModel)
build_headers             = zen(HeadersModel)
build_messages            = zen(MessagesModel)
build_message_types       = zen(MessageTypesModel)
build_presets             = zen(PresetsModel)
build_prompts             = zen(PromptsModel)
build_sections            = zen(SectionsModel)
build_status              = zen(StatusModel)
build_system              = zen(SystemModel)
build_theme               = zen(ThemeModel)
build_tips                = zen(TipsModel)
build_training_components = zen(TrainingComponentsModel)
build_ui                  = zen(UIModel)
build_validation          = zen(ValidationModel)
build_wandb_display       = zen(WandbDisplayModel)