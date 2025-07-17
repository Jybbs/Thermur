"""
User interaction and preset configuration schemas for the Thermur CLI.

This module contains models for interactive prompts and pre-configured training
presets that guide users through the CLI experience.
"""
from pydantic import BaseModel, DirectoryPath, Field


class DownloadModel(BaseModel, extra="forbid"):
    """
    Download and management configuration for the CLI.
    
    This model contains settings for file downloads, display options, and
    caching behavior, separate from the dataset schema used for training.
    """
    cache_dir: DirectoryPath = Field(
        default     = "data/cache",
        description = "Local directory for caching downloaded files."
    )
    globus_client_id: str = Field(
        default     = "ac349f52-8197-4a41-8d6d-5ae1c879273f",
        description = "Native app client ID for Globus OAuth2 authentication flow."
    )
    globus_dataset_path: str = Field(
        default     = "/1/published/publication_309/submitted_data",
        description = "Path to the WRF-Fire dataset within the FRDR Globus endpoint."
    )
    globus_endpoint_id: str = Field(
        default     = "f163c1b3-9c88-42f6-a7bb-5839ed6c4063",
        description = "UUID of the FRDR Globus endpoint hosting WRF-Fire simulations."
    )
    globus_scopes: str = Field(
        default     = "urn:globus:auth:scope:transfer.api.globus.org:all",
        description = "OAuth2 scopes required for Globus transfer operations."
    )


class PresetsModel(BaseModel, extra="forbid"):
    """
    Collection of all available training presets.
    
    Each preset provides a pre-configured set of parameters optimized for different
    training scenarios and use cases. The structure uses a dictionary format for
    easier access and maintenance.
    """
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
