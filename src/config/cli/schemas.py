"""
Simplified CLI configuration schemas.

This module provides configuration models for CLI components that need to be
configurable or contain structured data (arrays/dicts). Single-use strings
have been moved inline to reduce indirection.
"""
from pathlib           import Path
from pydantic          import BaseModel, computed_field, Field, PositiveInt, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing            import Literal

SECRETS_DIR = Path.home() / ".config" / "thermur" / "secrets"


class DisplayModel(BaseModel, extra="forbid"):
    """
    Display configuration for structured UI elements.
    
    Contains arrays and dictionaries that define the CLI's visual structure
    and may need to be extended or modified.
    """
    commands_available: list[dict[str, str]] = Field(
        default = [
            {
                "desc" : "Download WRF-Fire simulation datasets",
                "icon" : "📥",
                "name" : "download",
            },
            {
                "desc" : "Display system and configuration details",
                "icon" : "📋",
                "name" : "info",
            },
            {
                "desc" : "Monitor training progress and resources",
                "icon" : "🎨",
                "name" : "monitor",
            },
            {
                "desc" : "Explore and manage training runs",
                "icon" : "🏃",
                "name" : "runs",
            },
            {
                "desc" : "Train thermal drone flock behaviors",
                "icon" : "🚀",
                "name" : "train",
            },
            {
                "desc" : "Validate configuration and dependencies",
                "icon" : "✅",
                "name" : "validate",
            },
        ],
        description = "List of available commands with metadata for the main help display."
    )
    commands_examples: list[dict[str, str]] = Field(
        default = [
            {
                "command" : "thermur train",
                "desc"    : "Start interactive training",
                "note"    : "Guides you through configuration"
            },
            {
                "command" : "thermur train lightning.optimizer.learning_rate=0.001",
                "desc"    : "Custom learning rate",
                "note"    : "Override specific parameters"
            },
            {
                "command" : "thermur runs",
                "desc"    : "List recent training runs",
                "note"    : "View run history with status"
            },
            {
                "command" : "thermur runs show",
                "desc"    : "Show last run configuration",
                "note"    : "Paginated view of all settings"
            },
            {
                "command" : "thermur monitor",
                "desc"    : "Monitor training progress",
                "note"    : "Real-time wandb dashboard"
            },
            {
                "command" : "thermur validate",
                "desc"    : "Validate configuration",
                "note"    : "Check setup before training"
            },
            {
                "command" : "thermur train controller.expert.agent_count=20 lightning.optimizer.seed=42",
                "desc"    : "Train with overrides",
                "note"    : "Multiple parameters"
            },
            {
                "command" : "thermur download",
                "desc"    : "Download training data",
                "note"    : "Interactive selection of datasets"
            },
            {
                "command" : "thermur download -s",
                "desc"    : "Download sample data",
                "note"    : "Quick start with sample dataset"
            },
        ],
        description = "Example commands demonstrating common usage patterns for new users."
    )
    fire_gradient: list[str] = Field(
        default = [
            "red3", 
            "red1", 
            "orange_red1", 
            "dark_orange", 
            "orange1", 
            "yellow1", 
            "bright_yellow"
        ],
        description = (
            "Sequential color gradient used for animating fire intensity effects in "
            "the Thermur branding display."
        )
    )
    message_types: dict[str, dict[str, str]] = Field(
        default = {
            'config'  : {'icon': '⚙️ ', 'style': 'config'},
            'debug'   : {'icon': '🐛',  'style': 'debug'},
            'error'   : {'icon': '❌',  'style': 'error'},
            'flock'   : {'icon': '🪿',  'style': 'flock'},
            'info'    : {'icon': 'ℹ️ ', 'style': 'info'},
            'success' : {'icon': '✅',  'style': 'success'},
            'thermal' : {'icon': '🔥',  'style': 'thermal'},
            'warning' : {'icon': '⚠️ ', 'style': 'warning'},
        },
        description = "Message type configurations mapping semantic types to icons and style names."
    )
    override_examples: str = Field(
        default = (
            "# Override examples:\n"
            "lightning.optimizer.learning_rate=0.001   # Learning rate\n"
            "lightning.experience.batch_size=64        # Batch size\n"
            "controller.expert.agent_count=10          # Number of agents\n"
            "lightning.trainer.precision=32-true       # Training precision\n"
            "cli.wandb.mode=offline                    # W&B logging mode"
        ),
        description = "Multi-line help text showing example Hydra-style configuration overrides."
    )
    progress_bar_length: PositiveInt = Field(
        default     = 20,
        description = "Character width for rendering progress bars in terminal output."
    )
    questionary_style: dict[str, str] = Field(
        default = {
            'answer'      : 'fg:#4ecdc4 bold',
            'disabled'    : 'fg:#808080 italic',
            'highlighted' : 'fg:#ff6b6b bold',
            'instruction' : 'fg:#f38181',
            'pointer'     : 'fg:#ffe66d bold',
            'question'    : 'fg:#ff6b6b bold',
            'selected'    : 'fg:#4ecdc4',
            'separator'   : 'fg:#95e1d3',
            'text'        : 'fg:#ffffff',
        },
        description = "Color and style configuration for interactive questionary prompts."
    )
    styles: dict[str, str] = Field(
        default = {
            'accent'    : 'bright_magenta',
            'bright'    : 'bright_white',
            'config'    : 'bright_blue',
            'debug'     : 'italic grey70',
            'dim'       : 'grey30',
            'drone'     : 'yellow',
            'error'     : 'bold red',
            'flock'     : 'bold bright_cyan',
            'highlight' : 'on dark_blue',
            'info'      : 'cyan',
            'muted'     : 'grey50',
            'success'   : 'bold green',
            'thermal'   : 'bold orange_red1',
            'warning'   : 'bold yellow',
        },
        description = "Named Rich styles mapping semantic names to color and formatting definitions."
    )
    system_components: dict[str, str] = Field(
        default = {
            'cuda'     : '🎮 CUDA',
            'dataset'  : '📥 Dataset',
            'disk'     : '💿 Disk',
            'gpu'      : '💎 GPU',
            'memory'   : '💾 Memory',
            'mujoco'   : '🥽 MuJoCo',
            'platform' : '🖥️  Platform',
            'python'   : '🐍 Python',
            'thermur'  : '🔥 Thermur',
            'torch'    : '⚡ PyTorch',
        },
        description = "Display names with emoji icons for system components shown in info command."
    )
    system_table_columns: list[dict[str, str | int]] = Field(
        default = [
            {
                "header" : "Component",
                "style"  : "bright_cyan",
                "width"  : 15
            },
            {
                "header" : "Status",
                "style"  : "bright_white",
                "width"  : 40
            },
        ],
        description = "Column definitions including headers, styles, and widths for system info table."
    )
    training_component_configs: list[tuple[str, str, str]] = Field(
        default = [
            ("datamodule", "_system.datamodule", "📦 Data Module"),
            ("policy",     "_system.policy",     "🧠 Learning Policy"),
            ("trainer",    "_system.trainer",    "⚡ Lightning Trainer"),
        ],
        description = (
            "Component initialization tuples containing configuration key, hydra path, "
            "and display name for training setup workflow."
        )
    )


class DownloadModel(BaseModel, extra="forbid"):
    """
    Download and management configuration for the CLI.
    
    This model contains settings for file downloads, display options, and
    caching behavior, separate from the dataset schema used for training.
    """
    globus_client_id: str = Field(
        default     = "ac349f52-8197-4a41-8d6d-5ae1c879273f",
        description = (
            "Native application client identifier used for Globus OAuth2 authentication "
            "and authorization workflow."
        )
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
        description = (
            "OAuth2 permission scopes required for authorizing Globus file transfer "
            "operations and endpoint access."
        )
    )
    recommended_files: list[dict[str, str]] = Field(
        default = [
            {
                "desc" : "Light wind (3m/s) over short grass. Represents prescribed burns "
                         "or early-season fires in grasslands with stable atmospheric conditions.",
                "file" : "wrfout_W3F1R0",
            },
            {
                "desc" : "Moderate wind (5m/s) through brushy forest understory. Models typical "
                         "wildfire conditions with mixed vegetation and moderate atmospheric mixing.",
                "file" : "wrfout_W5F7R4",
            },
            {
                "desc" : "Strong wind (8m/s) through heavy dead trees and branches. Simulates "
                         "post-logging or storm damage areas with deep atmospheric mixing.",
                "file" : "wrfout_W8F13R6",
            },
            {
                "desc" : "Extreme wind (12m/s) in dense shrubland. Represents high-risk fire "
                         "weather in Mediterranean climates with strong temperature inversions.",
                "file" : "wrfout_W12F4R8",
            }
        ],
        description = (
            "Curated starter dataset files with detailed fire scenario descriptions "
            "for different atmospheric and vegetation conditions."
        )
    )
    sample_data_path: Path = Field(
        default     = Path("data/samples/wrf_sample.nc"),
        description = (
            "Filesystem path where extracted sample NetCDF dataset file will be saved "
            "after download and decompression."
        )
    )
    sample_data_url: str = Field(
        default     = "https://huggingface.co/datasets/Jybbs/sfire-samples/resolve/main/samples.tar.gz",
        description = (
            "Direct download URL pointing to compressed sample dataset archive hosted "
            "on Hugging Face model hub."
        )
    )
    sample_extract_dir: Path = Field(
        default     = Path("data"),
        description = (
            "Target directory for extracting downloaded sample dataset archive files "
            "during the download process."
        )
    )
    source: Literal["sample", "wrf-sfire", ""] = Field(
        default     = "",
        description = (
            "Selection of data source with 'sample' for quick start tutorial dataset "
            "or 'wrf-sfire' for complete research dataset."
        )
    )
    transfer_timeout: int = Field(
        default     = 86400,
        description = (
            "Maximum time in seconds to wait for Globus transfer completion before "
            "timing out with 24-hour default."
        )
    )
    wrf_sfire_dir: Path = Field(
        default     = Path("data/wrf-sfire"),
        description = (
            "Local filesystem directory designated for storing downloaded WRF-SFIRE "
            "simulation dataset files from Globus transfers."
        )
    )


class GlobusSecrets(BaseSettings):
    """
    Secure storage for Globus OAuth2 tokens.
    
    Uses Pydantic's BaseSettings with secrets_dir for automatic persistence.
    Each token field is stored as a separate file in the secrets directory,
    with the filename matching the field name.
    """
    refresh_token: SecretStr | None = Field(
        default     = None,
        description = (
            "Long-lived OAuth2 refresh token used to obtain new access tokens "
            "for Globus transfer operations."
        )
    )
    scope: str | None = Field(
        default     = None,
        description = (
            "Space-delimited list of OAuth2 scopes granted by the authentication "
            "token for Globus API access."
        )
    )
    secrets_path: Path = Field(
        default     = SECRETS_DIR,
        description = (
            "Local filesystem directory used for persisting OAuth2 tokens and other "
            "sensitive authentication credentials."
        )
    )
    
    model_config = SettingsConfigDict(
        case_sensitive = False,
        secrets_dir    = str(SECRETS_DIR) if SECRETS_DIR.exists() else None
    )
    
    @computed_field
    @property
    def is_valid(self) -> bool:
        """
        Check if all required token fields are present.
        """
        return all(
            getattr(self, field) is not None 
            for field in ['refresh_token', 'scope']
        )


class WandbConfig(BaseModel, extra="forbid"):
    """
    W&B configuration for training runs.
    
    Used by both CLI display and the imitation training pipeline
    for configuring Weights & Biases experiment tracking.
    """
    log_model: Literal["all", "best", "none"] = Field(
        default     = "all",
        description = "When to save model checkpoints to W&B"
    )
    mode: Literal["online", "offline", "disabled"] = Field(
        default     = "online",
        description = "W&B logging mode for training runs"
    )
    project: str = Field(
        default     = "thermur-imitation",
        description = "W&B project name for organizing experiments"
    )



