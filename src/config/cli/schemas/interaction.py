"""
User interaction and preset configuration schemas for the Thermur CLI.

This module contains models for interactive prompts and pre-configured training
presets that guide users through the CLI experience.
"""
from pathlib  import Path
from pydantic import BaseModel, Field
from typing   import Literal


class DownloadModel(BaseModel, extra="forbid"):
    """
    Download and management configuration for the CLI.
    
    This model contains settings for file downloads, display options, and
    caching behavior, separate from the dataset schema used for training.
    """
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
    recommended_files: list[dict[str, str]] = Field(
        default = [
            {
                "file" : "wrfout_W3F1R0",
                "desc" : "Light wind (3m/s) over short grass. Represents prescribed burns "
                         "or early-season fires in grasslands with stable atmospheric conditions."
            },
            {
                "file" : "wrfout_W5F7R4",  
                "desc" : "Moderate wind (5m/s) through brushy forest understory. Models typical "
                         "wildfire conditions with mixed vegetation and moderate atmospheric mixing."
            },
            {
                "file" : "wrfout_W8F13R6",
                "desc" : "Strong wind (8m/s) through heavy dead trees and branches. Simulates "
                         "post-logging or storm damage areas with deep atmospheric mixing."
            },
            {
                "file" : "wrfout_W12F4R8",
                "desc" : "Extreme wind (12m/s) in dense shrubland. Represents high-risk fire "
                         "weather in Mediterranean climates with strong temperature inversions."
            }
        ],
        description = "Recommended starter files with condition descriptions."
    )
    sample_data_path: Path = Field(
        default     = Path("data/samples/wrf_sample.nc"),
        description = "Local path where sample NetCDF file will be stored after extraction."
    )
    sample_data_url: str = Field(
        default     = "https://huggingface.co/datasets/Jybbs/sfire-samples/resolve/main/samples.tar.gz",
        description = "Hugging Face direct download URL for sample data tar.gz file."
    )
    sample_extract_dir: Path = Field(
        default     = Path("data"),
        description = "Directory where sample tar.gz will be extracted."
    )
    source: Literal["sample", "wrf-sfire", ""] = Field(
        default     = "",
        description = "Data source to download: 'sample' for quick start, 'wrf-sfire' for full dataset."
    )
    transfer_timeout: int = Field(
        default     = 86400,
        description = "Maximum seconds to wait for transfer completion (default: 24 hours)."
    )
    wrf_sfire_dir: Path = Field(
        default     = Path("data/wrf-sfire"),
        description = "Local directory for storing WRF-SFIRE dataset files from Globus."
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
