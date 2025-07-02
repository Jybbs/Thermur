"""
Training preset and tips schemas for the Thermur CLI.

This module defines models for pre-configured training presets and
helpful tips displayed to users.
"""
from pydantic import BaseModel, Field


class PresetModel(BaseModel, extra="forbid"):
    """
    Training preset configuration.
    
    Defines a named collection of parameters optimized for a specific
    use case, from rapid debugging to full-scale training runs.
    """
    best_for: str = Field(
        description = "Use case this preset is optimized for"
    )
    desc: str = Field(
        description = "Brief description of the preset"
    )
    emoji: str = Field(
        description = "Emoji icon for the preset"
    )
    name: str = Field(
        description = "Preset identifier name"
    )
    prompt: str = Field(
        description = "Display text for preset selection"
    )


class PresetsModel(BaseModel, extra="forbid"):
    """
    Collection of all available training presets.
    
    Each preset provides a pre-configured set of parameters optimized
    for different training scenarios and use cases.
    """
    custom: PresetModel = Field(
        default = PresetModel(
            best_for = "Advanced users",
            desc     = "Start from scratch with full control",
            emoji    = "🧵",
            name     = "custom",
            prompt   = "🧵 custom    - Configure everything manually"
        ),
        description = "Custom configuration preset"
    )
    debug: PresetModel = Field(
        default = PresetModel(
            best_for = "Troubleshooting issues",
            desc     = "Verbose logging & validation checks",
            emoji    = "🔍",
            name     = "debug",
            prompt   = "🔍 debug     - Detailed diagnostics"
        ),
        description = "Debug configuration preset"
    )
    large: PresetModel = Field(
        default = PresetModel(
            best_for = "Production & final models",
            desc     = "High-capacity models & longer training",
            emoji    = "💪",
            name     = "large",
            prompt   = "💪 large     - Maximum capacity"
        ),
        description = "Large-scale configuration preset"
    )
    quick: PresetModel = Field(
        default = PresetModel(
            best_for = "Quick experiments & debugging",
            desc     = "Minimal setup for rapid testing",
            emoji    = "⚡",
            name     = "quick",
            prompt   = "⚡ quick     - Fast testing & experiments"
        ),
        description = "Quick test configuration preset"
    )
    standard: PresetModel = Field(
        default = PresetModel(
            best_for = "Regular training runs",
            desc     = "Balanced configuration for most tasks",
            emoji    = "🔥",
            name     = "standard",
            prompt   = "🔥 standard  - Balanced performance"
        ),
        description = "Standard configuration preset"
    )
    table_title: str = Field(
        default     = "Available Presets",
        description = "Title for presets table display"
    )


class TipsModel(BaseModel, extra="forbid"):
    """
    Training tips and guidance for CLI users.
    
    Provides quick, actionable advice for using the CLI effectively,
    displayed in the info command to help users get started.
    """
    bullet_style: str = Field(
        default     = "bright_yellow",
        description = "Style for tip bullet points"
    )
    section_style: str = Field(
        default     = "bright_yellow",
        description = "Style for tips section"
    )
    section_title: str = Field(
        default     = "💡 Quick Tips",
        description = "Title for tips section"
    )
    tips: list[dict[str, str]] = Field(
        default = [
            {
                "desc"    : "Use presets for quick starts",
                "command" : "thermur train --preset quick",
            },
            {
                "desc"    : "Monitor training live",
                "command" : "thermur monitor --project my-experiment",
            },
            {
                "desc"    : "Override any parameter",
                "command" : "thermur train --config hyperparameters.lr=0.001",
            },
        ],
        description = "List of training tips with commands"
    )