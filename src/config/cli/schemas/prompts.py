"""
Interactive prompts and presets configuration schemas.

This module defines configuration for user interaction, prompt styling,
and training presets used by the CLIPrompts helper.
"""
from pydantic import BaseModel, Field


class PromptsModel(BaseModel, extra="forbid"):
    """
    Interactive prompt and dialog configuration.
    
    This model contains all settings for user interaction including error messages,
    prompt templates, and questionary styling. It manages the conversational flow
    of the CLI application.
    """
    config_not_found_error: str = Field(
        default     = "Configuration '{config}' not found in available workloads",
        description = "Error message template for missing configurations."
    )
    confirm_override_prompt: str = Field(
        default     = "Apply {count} override(s)?",
        description = "Confirmation prompt for applying overrides."
    )
    override_prompt: str = Field(
        default     = "Enter override value for {field_name}:",
        description = "Prompt template for configuration overrides."
    )
    package_missing_error: str = Field(
        default     = "Required package '{package}' is not installed",
        description = "Error message template for missing packages."
    )
    presets: dict[str, dict[str, str]] = Field(
        default = {
            "quick": {
                "best_for" : "Quick experiments & debugging",
                "desc"     : "Minimal setup for rapid testing",
                "emoji"    : "⚡",
                "name"     : "quick"
            },
            "standard": {
                "best_for" : "Regular training runs",
                "desc"     : "Balanced configuration for most tasks",
                "emoji"    : "🔥",
                "name"     : "standard"
            },
            "large": {
                "best_for" : "Production & final models",
                "desc"     : "High-capacity models & longer training",
                "emoji"    : "💪",
                "name"     : "large"
            },
            "debug": {
                "best_for" : "Troubleshooting issues",
                "desc"     : "Verbose logging & validation checks",
                "emoji"    : "🔍",
                "name"     : "debug"
            },
            "custom": {
                "best_for" : "Advanced users",
                "desc"     : "Start from scratch with full control",
                "emoji"    : "🧵",
                "name"     : "custom"
            }
        },
        description = (
            "Training preset configurations optimized for different use cases. "
            "Each preset includes display metadata and configuration details."
        )
    )
    presets_table_columns: list[tuple[str, str, int, str]] = Field(
        default = [
            ("Preset",      "bright_cyan", 15, "left"),
            ("Description", "white",       40, "left"),
            ("Best For",    "grey70",      30, "left")
        ],
        description = "Column configuration for presets selection table."
    )
    python_version_error: str = Field(
        default     = "Python {current} detected, but {required} or higher is required",
        description = "Error message template for Python version mismatch."
    )
    questionary_style: list[tuple[str, str]] = Field(
        default = [
            ('question',    'fg:#ff6b6b bold'),
            ('answer',      'fg:#4ecdc4 bold'),
            ('pointer',     'fg:#ffe66d bold'),
            ('highlighted', 'fg:#ff6b6b bold'),
            ('selected',    'fg:#4ecdc4'),
            ('separator',   'fg:#95e1d3'),
            ('instruction', 'fg:#f38181'),
            ('text',        'fg:#ffffff'),
            ('disabled',    'fg:#808080 italic'),
        ],
        description = "Questionary prompt styling configuration."
    )
    workload_selection_prompt: str = Field(
        default     = "Select a workload configuration:",
        description = "Prompt shown when user needs to select a workload."
    )