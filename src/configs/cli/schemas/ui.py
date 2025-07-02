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
    default_badge_style: str = Field(
        default     = "success",
        description = "Default style for badges"
    )
    default_section_style: str = Field(
        default     = "accent",
        description = "Default style for section headers"
    )
    features_list: list[dict[str, str]] = Field(
        default = [
            {
                "name" : "🔥 Thermal Constraints",
                "desc" : "Physics-based heat modeling for drone safety"
            },
            {
                "name" : "💫 Flock Coordination",
                "desc" : "Multi-agent flocking with separation and alignment"
            },
            {
                "name" : "🧠 Imitation Learning",
                "desc" : "Expert policy cloning with behavioral cloning"
            },
            {
                "name" : "🎨 wandb Integration",
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
                "width"  : 28,
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
    message_types: dict[str, dict[str, str]] = Field(
        default = {
            "info": {
                "icon"  : "ℹ️ ",
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
                "icon"  : "💫",
                "style" : "flock"
            },
            "config": {
                "icon"  : "🔎",
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
    progress_bar_length: PositiveInt = Field(
        default     = 20,
        description = "Length for progress bars"
    )
    progress_style: str = Field(
        default     = "thermal",
        description = "Style for progress bars"
    )
    progress_unfilled_color: str = Field(
        default     = "grey30",
        description = "Color for unfilled progress"
    )
    resource_details_template: str = Field(
        default     = "[white]{:.1f}{} free of {:.1f}{}[/]",
        description = "Template for resource details display"
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
