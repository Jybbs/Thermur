"""
Pydantic schemas for the Thermur CLI configuration.

This package organizes CLI-related configuration models by functionality,
providing type-safe validation for all CLI components.
"""
from __future__ import annotations

from .core     import CLIModel, CommandsModel, TrainingComponentsModel
from .explorer import ExplorerMessagesModel, ExplorerModel
from .messages import (
    HeadersModel,
    MessagesModel,
    MessageTypesModel,
    PromptsModel,
    SectionsModel,
    StatusModel,
)
from .presets  import PresetModel, PresetsModel, TipsModel
from .system   import SystemModel, ValidationModel, WandbDisplayModel
from .ui       import ThemeModel, UIModel

__all__ = [
    # .core
    "CLIModel",
    "CommandsModel", 
    "TrainingComponentsModel",
    
    # .explorer
    "ExplorerMessagesModel",
    "ExplorerModel",
    
    # .messages
    "HeadersModel",
    "MessagesModel",
    "MessageTypesModel",
    "PromptsModel",
    "SectionsModel",
    "StatusModel",
    
    # .presets
    "PresetModel",
    "PresetsModel",
    "TipsModel",
    
    # .system
    "SystemModel",
    "ValidationModel",
    "WandbDisplayModel",
    
    # .ui
    "ThemeModel",
    "UIModel",
]