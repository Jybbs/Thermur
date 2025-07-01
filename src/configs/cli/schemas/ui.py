"""
UI and theme configuration schemas for the Thermur CLI.

This module defines models for terminal rendering, styling, and visual
elements used throughout the CLI interface.
"""
from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt


class ThemeModel(BaseModel, extra="forbid"):
    """
    Thermal physics-inspired styling for CLI interface.
    
    Color mappings reflecting thermal domain physics. The fire gradient
    represents temperature transitions from T_min to T_max while
    maintaining terminal readability.
    """
    fire_gradient: list[str] = Field(
        default = [
            "#8B0000",
            "#DC143C",
            "#FF4500",
            "#FF8C00",
            "#FFD700",
            "#FFFF00",
            "#FFFACD",
        ],
        description = "Color gradient representing thermal transitions"
    )
    styles: dict[str, str] = Field(
        default = {
            "accent"    : "bold bright_cyan",
            "dim"       : "grey50",
            "drone"     : "bright_magenta",
            "error"     : "bold bright_red",
            "heat"      : "bright_red on grey23",
            "highlight" : "bold bright_blue",
            "info"      : "bright_cyan",
            "muted"     : "grey70",
            "success"   : "bold bright_green",
            "flock"     : "bright_blue",
            "thermal"   : "bold red",
            "warning"   : "bright_yellow",
        },
        description = "Named style mappings for consistent theming"
    )


class UIModel(BaseModel, extra="forbid"):
    """
    User interface constants for Rich components.
    
    Defines static configuration for all Rich-rendered components, such
    as padding, border styles, colors, and character sets.
    """
    bullet_char: str = Field(
        default     = "•",
        description = "Character for bullet points"
    )
    command_style: str = Field(
        default     = "bold accent",
        description = "Style for command text"
    )
    default_badge_style: str = Field(
        default     = "success",
        description = "Default style for badges"
    )
    default_section_style: str = Field(
        default     = "accent",
        description = "Default style for section headers"
    )
    error_color: str = Field(
        default     = "red",
        description = "Color for errors and failures"
    )
    features_list: list[dict[str, str]] = Field(
        default = [
            {
                "name" : "🔥 Thermal Constraints",
                "desc" : "Physics-based heat modeling for drone safety"
            },
            {
                "name" : "🪽  Flock Coordination",
                "desc" : "Multi-agent flocking with obstacle avoidance"
            },
            {
                "name" : "🧠 Imitation Learning",
                "desc" : "Expert policy cloning with BC and DAgger"
            },
            {
                "name" : "🪄  wandb Integration",
                "desc" : "Real-time experiment tracking and visualization"
            },
            {
                "name" : "🎯 Hydra Configuration",
                "desc" : "Modular config system with validation"
            },
            {
                "name" : "📈 Live Visualization",
                "desc" : "3D rendering of flock dynamics"
            },
        ],
        description = "List of features to display in the features table"
    )
    features_table_columns: list[dict[str, str | int]] = Field(
        default = [
            {
                "header" : "Feature",
                "style"  : "bright_cyan",
                "width"  : 35,
                "align"  : "left"
            },
            {
                "header" : "Description",
                "style"  : "white",
                "width"  : 60,
                "align"  : "left"
            },
        ],
        description = "Column definitions for features table"
    )
    filled_char: str = Field(
        default     = "█",
        description = "Character for filled progress bars"
    )
    header_text_style: str = Field(
        default     = "bold bright_white",
        description = "Style for header text"
    )
    panel_border_style: str = Field(
        default     = "bright_blue",
        description = "Border style for Rich panels"
    )
    panel_box: str = Field(
        default     = "ROUNDED",
        description = "Box type for Rich panels"
    )
    panel_padding: tuple[PositiveInt, PositiveInt] = Field(
        default     = (1, 3),
        description = "Padding for Rich panels (vertical, horizontal)"
    )
    primary_color: str = Field(
        default     = "dodger_blue2",
        description = "Primary color for important elements"
    )
    progress_bar_length: PositiveInt = Field(
        default     = 20,
        description = "Length for progress bars"
    )
    progress_complete_style: str = Field(
        default     = "bright_red",
        description = "Style for completed progress"
    )
    progress_spinner: str = Field(
        default     = "dots",
        description = "Spinner type for progress indicators"
    )
    progress_style: str = Field(
        default     = "thermal",
        description = "Style for progress bars"
    )
    progress_unfilled_color: str = Field(
        default     = "grey30",
        description = "Color for unfilled progress"
    )
    resource_color_critical: str = Field(
        default     = "red",
        description = "Color for critical resource status"
    )
    resource_color_good: str = Field(
        default     = "bright_green",
        description = "Color for good resource status"
    )
    resource_color_warning: str = Field(
        default     = "yellow",
        description = "Color for warning resource status"
    )
    resource_details_template: str = Field(
        default     = "[white]{:.1f}{} free of {:.1f}{}[/]",
        description = "Template for resource details display"
    )
    secondary_color: str = Field(
        default     = "grey70",
        description = "Secondary color for less prominent text"
    )
    subtitle_text_style: str = Field(
        default     = "muted italic",
        description = "Style for subtitle text"
    )
    success_color: str = Field(
        default     = "green",
        description = "Color for success messages"
    )
    syntax_theme: str = Field(
        default     = "monokai",
        description = "Theme for syntax highlighting"
    )
    system_table_columns: list[dict[str, str | int]] = Field(
        default = [
            {
                "header" : "Component",
                "style"  : "bright_cyan",
                "width"  : 20
            },
            {
                "header" : "Value",
                "style"  : "white",
                "width"  : 50
            },
        ],
        description = "Column definitions for system info table"
    )
    system_table_settings: dict[str, str | bool] = Field(
        default = {
            "box"          : "MINIMAL",
            "border_style" : "bright_blue",
            "title_style"  : "bold bright_cyan",
            "show_lines"   : True,
        },
        description = "Settings for system info table"
    )
    system_components: dict[str, str] = Field(
        default = {
            "python"       : "Python",
            "torch"        : "PyTorch",
            "cuda"         : "CUDA Available",
            "device_count" : "GPU Devices",
            "platform"     : "Platform",
            "memory"       : "Memory (RAM)",
            "disk"         : "Disk Storage",
        },
        description = "Component names for system info display"
    )
    system_logic: dict[str, dict[str, str | bool]] = Field(
        default = {
            "python" : {
                "format" : "v{}",
                "key"    : "python"
            },
            "torch" : {
                "format" : "v{}",
                "key"    : "torch"
            },
            "cuda" : {
                "format" : "{}",
                "key"    : "cuda"
            },
            "device_count" : {
                "format" : "{} device(s)",
                "key"    : "device_count"
            },
            "platform" : {
                "format" : "{}",
                "key"    : "platform"
            },
            "memory" : {
                "is_resource" : True,
                "available"   : "memory_available",
                "total"       : "memory_total",
                "unit"        : "GB"
            },
            "disk" : {
                "is_resource" : True,
                "available"   : "disk_available",
                "total"       : "disk_total",
                "unit"        : "GB"
            },
        },
        description = "Logic for formatting system component values"
    )
    table_border_style: str = Field(
        default     = "bright_blue",
        description = "Border style for Rich tables"
    )
    table_box: str = Field(
        default     = "MINIMAL",
        description = "Box type for Rich tables"
    )
    table_header_style: str = Field(
        default     = "bold bright_blue",
        description = "Style for table headers"
    )
    table_padding: tuple[NonNegativeInt, PositiveInt] = Field(
        default     = (0, 1),
        description = "Padding for Rich tables (vertical, horizontal)"
    )
    table_title_style: str = Field(
        default     = "bold bright_cyan",
        description = "Style for table titles"
    )
    title_text_style: str = Field(
        default     = "bold bright_cyan",
        description = "Style for title text"
    )
    unfilled_char: str = Field(
        default     = "░",
        description = "Character for unfilled progress bars"
    )
    wandb_url_placeholder: str = Field(
        default     = "YOUR_USERNAME",
        description = "Placeholder for wandb URLs"
    )
    warning_color: str = Field(
        default     = "yellow",
        description = "Color for warnings"
    )
    message_types: dict[str, dict[str, str]] = Field(
        default = {
            "info": {
                "icon"  : "ℹ️",
                "style" : "info"
            },
            "success": {
                "icon"  : "✅",
                "style" : "success"
            },
            "warning": {
                "icon"  : "⚠️",
                "style" : "warning"
            },
            "error": {
                "icon"  : "❌",
                "style" : "error"
            },
            "thermal": {
                "icon"  : "🔥",
                "style" : "thermal"
            },
            "flock": {
                "icon"  : "🪽",
                "style" : "flock"
            },
            "config": {
                "icon"  : "⚙️",
                "style" : "accent"
            },
            "tip": {
                "icon"  : "💡",
                "style" : "muted"
            },
            "standard": {
                "icon"  : "",
                "style" : "white"
            },
        },
        description = "Message type configurations for print_message"
    )
