"""
Rich console configuration and utilities for the Thermur CLI.

This module provides a consistent console interface leveraging Rich's
built-in styling and formatting capabilities.
"""
from rich.align    import Align
from rich.box      import MINIMAL, ROUNDED
from rich.console  import Console
from rich.layout   import Layout
from rich.panel    import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule     import Rule
from rich.table    import Table
from rich.text     import Text
from rich.theme    import Theme

# Thermal-inspired theme configuration
theme = Theme({
    "info"      : "bright_cyan",
    "warning"   : "bright_yellow", 
    "error"     : "bold bright_red",
    "success"   : "bold bright_green",
    "highlight" : "bold bright_blue",
    "muted"     : "grey70",
    "dim"       : "grey50",
    "thermal"   : "bold red",
    "heat"      : "bright_red on grey23",
    "drone"     : "bright_magenta",
    "swarm"     : "bright_blue",
    "accent"    : "bold bright_cyan",
})

# Global console instance with theme
console = Console(theme=theme, highlight=False)

# Message type configurations
MESSAGE_STYLES = {
    "step"    : {"icon": "🔥",  "style": "thermal"},
    "info"    : {"icon": "💡",  "style": "info"},
    "warning" : {"icon": "🌡️", "style": "warning"},
    "error"   : {"icon": "🚨",  "style": "error"},
    "success" : {"icon": "✅",  "style": "success"},
    "swarm"   : {"icon": "🐦‍⬛", "style": "swarm"},
    "thermal" : {"icon": "🔥",  "style": "thermal"},
    "tip"     : {"icon": "💭",  "style": "muted"},
    "config"  : {"icon": "⚙️",  "style": "accent"},
}


def create_progress() -> Progress:
    """
    Create a thermal-themed progress bar.
    
    Returns:
        Configured Progress instance with thermal styling
    """
    return Progress(
        SpinnerColumn(spinner_name="dots", style="thermal"),
        TextColumn("[thermal]{task.description}[/thermal]"),
        BarColumn(bar_width=30, style="thermal", complete_style="bright_red"),
        TaskProgressColumn(),
        "•",
        TimeElapsedColumn(),
        "•",
        MofNCompleteColumn(),
        console         = console,
        expand          = False,
        transient       = False,
    )


def print_header(
    title    : str,
    subtitle : str | None = None
):
    """
    Print a styled header panel.
    
    Creates a visually distinct panel with the provided title, helping to
    organize the CLI output into logical sections for better readability.
    
    Args:
        title    : Main header text
        subtitle : Optional subtitle text
    """
    console.print()
    
    title_text = Text()
    
    # Apply fire gradient only to "Thermur" if it's in the title
    if "Thermur" in title:
        parts = title.split("Thermur", 1)
        
        # Add pre-Thermur text if any
        if parts[0]:
            title_text.append(parts[0], style="bold bright_white")
        
        # Add "Thermur" with clean fire gradient
        thermur_text = "Thermur"
        # Clean gradient: dark red → crimson → orange → gold → yellow → white
        fire_colors = [
            "bold #8B0000",      # T - dark red
            "bold #DC143C",      # h - crimson
            "bold #FF4500",      # e - orange red
            "bold #FF8C00",      # r - dark orange
            "bold #FFD700",      # m - gold
            "bold #FFFF00",      # u - yellow
            "bold #FFFACD",      # r - lemon chiffon (almost white)
        ]
        
        for i, char in enumerate(thermur_text):
            color = fire_colors[i]
            title_text.append(char, style=color)
        
        # Add post-Thermur text if any
        if parts[1]:
            title_text.append(parts[1], style="bold bright_white")
    else:
        # For non-Thermur titles, use bright cyan
        title_text.append(title, style="bold bright_cyan")
    
    if subtitle:
        title_text.append("\n")
        title_text.append(subtitle, style="muted italic")
    
    panel = Panel(
        Align.center(title_text),
        border_style = "bright_blue",
        box          = ROUNDED,
        padding      = (1, 3),
    )
    
    console.print(panel)
    console.print()


def print_section(title: str, style: str = "accent"):
    """
    Print a section divider with title.
    
    Creates a visual separator between different sections of output.
    
    Args:
        title : Section title
        style : Style to apply to the rule
    """
    console.print()
    console.print(Rule(f" {title} ", style=style))
    console.print()


def print_message(
    message : str,
    msg_type: str = "info"
):
    """
    Print a styled message with appropriate icon and formatting.
    
    This function provides a unified interface for all message types,
    reducing code duplication while maintaining consistent styling.
    
    Args:
        message  : The message text to display
        msg_type : Type of message (step, info, warning, error, success, swarm, thermal, tip, config)
    """
    config = MESSAGE_STYLES.get(msg_type, MESSAGE_STYLES["info"])
    console.print(f"[{config['style']}]{config['icon']} {message}[/{config['style']}]")


def print_command_example(
    description : str,
    command     : str,
    note        : str | None = None
):
    """
    Print a formatted command example.
    
    Shows command examples in a visually distinct way to help users
    understand how to use the CLI.
    
    Args:
        description : What the command does
        command     : The actual command to run
        note        : Optional note about the command
    """
    console.print(f"  [muted]{description}:[/muted]")
    console.print(f"  [bold accent]$ {command}[/bold accent]")
    if note:
        console.print(f"  [dim italic]  {note}[/dim italic]")
    console.print()


def print_config_value(
    key   : str,
    value : str,
    desc  : str | None = None
):
    """
    Print a configuration key-value pair.
    
    Formats configuration information in a consistent way.
    
    Args:
        key   : Configuration key
        value : Configuration value
        desc  : Optional description
    """
    if desc:
        console.print(f"  [accent]{key}[/accent] = [bright_white]{value}[/bright_white]  [dim]# {desc}[/dim]")
    else:
        console.print(f"  [accent]{key}[/accent] = [bright_white]{value}[/bright_white]")


def print_status_badge(
    label  : str,
    status : str,
    style  : str = "success"
):
    """
    Print a status badge.
    
    Creates a compact status indicator for various system states.
    
    Args:
        label  : Badge label
        status : Status text
        style  : Style to apply
    """
    badge = f"[{style}][ {label}: {status} ][/{style}]"
    console.print(badge)


def create_feature_table() -> Table:
    """
    Create a table showcasing Thermur features.
    
    Returns:
        A formatted table with feature information
    """
    table = Table(
        title        = "✨ Thermur Features",
        title_style  = "bold bright_cyan",
        header_style = "bold bright_blue",
        border_style = "bright_blue",
        box          = MINIMAL,
        show_edge    = False,
        padding      = (0, 1),
        expand       = False,
    )
    
    table.add_column("Feature",      style="bright_cyan",  width=25)
    table.add_column("Description",  style="bright_white", width=45)
    table.add_column("Status",       style="bright_green", width=12)
    
    features = [
        ("🔥 Thermal Constraints",    "Realistic heat modeling for drone swarms",         "✅ Ready"),
        ("🐦‍⬛ Swarm Intelligence",      "Multi-agent coordination and flocking",            "✅ Ready"),
        ("🎓 Imitation Learning",     "Learn from expert demonstrations",                 "✅ Ready"),
        ("📊 wandb Integration",      "Real-time experiment tracking",                    "✅ Ready"),
        ("🎮 GPU Acceleration",       "CUDA support for fast training",                   "✅ Ready"),
        ("🔧 Hydra Configuration",    "Flexible experiment configuration",                "✅ Ready"),
        ("📈 Live Visualization",     "Real-time swarm behavior rendering",               "🚧 Beta"),
    ]
    
    for feature, desc, status in features:
        table.add_row(feature, desc, status)
    
    return table


def print_wandb_info(
    project : str,
    url     : str | None = None
):
    """
    Print wandb project information.
    
    Displays wandb integration status and provides links to monitoring dashboards
    when available.
    
    Args:
        project : The wandb project name
        url     : Optional URL to the project dashboard
    """
    if url and "YOUR_USERNAME" not in url:
        console.print(f"[swarm]📊 Dashboard: [link={url}]{url}[/link][/swarm]")
    else:
        console.print(f"[swarm]📊 Project: [bright_cyan]{project}[/bright_cyan][/swarm]")


def print_training_tips():
    """
    Print helpful training tips.
    
    Shows useful information to help users get the most out of their
    training runs.
    """
    tips = [
        ("Use presets for quick starts", "thermur train --preset quick"),
        ("Monitor training live", "thermur monitor --project my-experiment"),
        ("Explore configurations", "thermur configure"),
        ("Override any parameter", "thermur train --config hyperparameters.lr=0.001"),
    ]
    
    print_section("💡 Quick Tips", style="bright_yellow")
    
    for tip, command in tips:
        console.print(f"  [bright_yellow]•[/bright_yellow] {tip}")
        console.print(f"    [dim]{command}[/dim]")
    
    console.print()
