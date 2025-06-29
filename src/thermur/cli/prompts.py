"""
Interactive prompts for the Thermur CLI with enhanced visual feedback.

This module provides rich user interaction functionality for gathering input,
confirming actions, and guiding users through configuration choices.
"""
import questionary

from rich.align  import Align
from rich.panel  import Panel
from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.table  import Table

from .ui import console, print_message, print_section


# Define a thermal-inspired style for questionary
thermal_style = questionary.Style([
    ('question',       'fg:#ff6b6b bold'),      # Bright red for questions
    ('answer',         'fg:#4ecdc4 bold'),      # Bright cyan for answers
    ('pointer',        'fg:#ffe66d bold'),      # Bright yellow for pointer
    ('highlighted',    'fg:#ff6b6b bold'),      # Bright red for highlighted
    ('selected',       'fg:#4ecdc4'),           # Cyan for selected
    ('separator',      'fg:#95e1d3'),           # Light cyan for separators
    ('instruction',    'fg:#f38181'),           # Light red for instructions
    ('text',           'fg:#ffffff'),           # White for text
    ('disabled',       'fg:#808080 italic'),    # Gray for disabled options
])


def select_configuration_preset() -> str | None:
    """
    Prompt user to select a configuration preset with visual descriptions.
    
    Returns:
        Selected preset name or None if user chooses custom configuration
    """
    print_section("Configuration Presets", "config")
    
    # Create a table showing preset details
    table = Table(
        show_header  = True,
        header_style = "bold bright_cyan",
        border_style = "bright_blue",
        title        = "Available Presets",
        title_style  = "bold bright_white",
        show_edge    = False,
        padding      = (0, 1),
    )
    
    table.add_column("Preset",       style="bright_cyan",   width=12)
    table.add_column("Description",  style="bright_white",  width=40)
    table.add_column("Best For",     style="bright_yellow", width=30)
    
    presets = [
        ("quick",    "Minimal setup for rapid testing",          "Quick experiments & debugging"),
        ("standard", "Balanced configuration for most tasks",     "Regular training runs"),
        ("large",    "High-capacity models & longer training",   "Production & final models"),
        ("debug",    "Verbose logging & validation checks",      "Troubleshooting issues"),
        ("custom",   "Start from scratch with full control",     "Advanced users"),
    ]
    
    for name, desc, use_case in presets:
        if name == "custom":
            table.add_row(f"[italic]{name}[/italic]", f"[italic]{desc}[/italic]", f"[italic]{use_case}[/italic]")
        else:
            table.add_row(name, desc, use_case)
    
    console.print(table)
    console.print()
    
    # Create choices with emojis
    choices = [
        questionary.Choice("⚡ quick     - Fast testing & experiments",                     value="quick"),
        questionary.Choice("🔥 standard  - Balanced performance",                          value="standard"),
        questionary.Choice("💪 large     - Maximum capacity",                              value="large"),
        questionary.Choice("🔍 debug     - Detailed diagnostics",                          value="debug"),
        questionary.Separator("─" * 40),
        questionary.Choice("🎨 custom    - Configure everything manually",                 value=None),
    ]
    
    preset = questionary.select(
        "Which configuration preset would you like to use?",
        choices = choices,
        style   = thermal_style,
    ).ask()
    
    if preset:
        print_message(f"Selected preset: [bright_cyan]{preset}[/bright_cyan]", "success")
    else:
        print_message("Custom configuration selected - full control mode", "info")
    
    return preset


def ask_wandb_project_name() -> str:
    """
    Prompt for wandb project name with suggestions.
    
    Returns:
        Project name for wandb tracking
    """
    console.print()
    print_message("Configure experiment tracking", "swarm")
    console.print("[muted]wandb will track metrics, logs, and model checkpoints[/muted]")
    console.print()
    
    # Show example project names
    examples = Panel(
        "[bright_cyan]Examples:[/bright_cyan]\n"
        "  • thermal-swarm-v1\n"
        "  • drone-flocking-experiments\n" 
        "  • heat-aware-navigation\n"
        "  • imitation-learning-tests",
        border_style = "bright_blue",
        padding      = (0, 2),
    )
    console.print(examples)
    console.print()
    
    project = questionary.text(
        "Enter wandb project name:",
        default     = "thermur",
        style       = thermal_style,
        instruction = "(press Enter for default)",
    ).ask()
    
    print_message(f"Project name: [bright_cyan]{project}[/bright_cyan]", "success")
    return project


def ask_for_config_overrides() -> list[str]:
    """
    Prompt for additional configuration overrides with examples.
    
    Returns:
        List of Hydra-style configuration overrides
    """
    console.print()
    print_section("Advanced Configuration", "config")
    
    # Show override syntax examples
    syntax_panel = Panel(
        Syntax(
            "# Override examples:\n"
            "hyperparameters.lr=0.001          # Learning rate\n"
            "hyperparameters.batch_size=64     # Batch size\n"
            "swarm.num_drones=10               # Number of drones\n"
            "environment.max_temp=85.0         # Temperature limit\n"
            "+experiment=my_custom_setup       # Load experiment",
            "python",
            theme        = "monokai",
            line_numbers = False,
        ),
        title        = "Configuration Override Syntax",
        border_style = "bright_blue",
        padding      = (1, 2),
    )
    console.print(syntax_panel)
    console.print()
    
    overrides = []
    
    add_overrides = questionary.confirm(
        "Would you like to add configuration overrides?",
        default = False,
        style   = thermal_style,
    ).ask()
    
    if add_overrides:
        console.print()
        console.print("[muted]Enter overrides one at a time (empty line to finish):[/muted]")
        
        while True:
            override = questionary.text(
                "Override:",
                style       = thermal_style,
                instruction = "(e.g., hyperparameters.lr=0.001)",
            ).ask()
            
            if not override:
                break
            
            overrides.append(override)
            console.print(f"  [success]✓[/success] Added: [bright_cyan]{override}[/bright_cyan]")
        
        if overrides:
            console.print()
            print_message(f"Added {len(overrides)} configuration override(s)", "success")
    
    return overrides


def confirm_system_override(issues: list[str]) -> bool:
    """
    Confirm whether to proceed despite system issues.
    
    Args:
        issues : List of system/configuration issues found
        
    Returns:
        True to proceed, False to cancel
    """
    console.print()
    warning_panel = Panel(
        "[bold bright_yellow]⚠️  Configuration Issues Detected[/bold bright_yellow]\n\n" +
        "\n".join(f"• {issue}" for issue in issues),
        border_style = "bright_yellow",
        padding      = (1, 2),
    )
    console.print(warning_panel)
    console.print()
    
    return Confirm.ask(
        "[bright_yellow]Do you want to proceed anyway?[/bright_yellow]",
        console = console,
        default = False,
    )


def show_training_summary(config: dict) -> bool:
    """
    Display training configuration summary and confirm.
    
    Args:
        config : Dictionary containing training configuration
        
    Returns:
        True to proceed with training, False to cancel
    """
    console.print()
    print_section("Training Configuration Summary", "thermal")
    
    # Create summary table
    table = Table(
        show_header  = False,
        border_style = "bright_blue",
        box          = None,
        padding      = (0, 2),
        expand       = False,
    )
    
    table.add_column("Setting",  style="bright_cyan",  width=20)
    table.add_column("Value",    style="bright_white", width=30)
    
    # Add configuration rows
    gpu_status = "🎮 GPU Acceleration" if config["gpu_available"] else "💻 CPU Mode"
    table.add_row("Configuration", f"[bright_yellow]{config['preset']}[/bright_yellow]")
    table.add_row("wandb Project", f"[bright_blue]{config['wandb_project']}[/bright_blue]")
    table.add_row("Overrides",     f"[bright_magenta]{config['overrides']} custom settings[/bright_magenta]")
    table.add_row("Hardware",      gpu_status)
    
    # Center the table
    console.print(Align.center(table))
    console.print()
    
    # Training readiness indicator
    ready_panel = Panel(
        Align.center(
            "[bold bright_green]✅ Ready to Train![/bold bright_green]\n"
            "[muted]Your thermal swarm is configured and ready to learn[/muted]",
            vertical="middle",
        ),
        border_style = "bright_green",
        padding      = (1, 3),
    )
    console.print(ready_panel)
    console.print()
    
    return Confirm.ask(
        "[bright_green]Start training with this configuration?[/bright_green]",
        console = console,
        default = True,
    )


def select_config_to_edit(configs: list[str]) -> str | None:
    """
    Select a configuration to edit from available options.
    
    Args:
        configs : List of available configuration names
        
    Returns:
        Selected configuration name or None
    """
    if not configs:
        print_message("No configurations available", "warning")
        return None
    
    # Group configs by category
    categories = {}
    for config in configs:
        if '.' in config:
            category = config.split('.')[0]
        else:
            category = "main"
        
        if category not in categories:
            categories[category] = []
        categories[category].append(config)
    
    # Build choices with categories
    choices = []
    for category, items in sorted(categories.items()):
        if category != "main":
            choices.append(questionary.Separator(f"── {category.title()} ──"))
        
        for item in sorted(items):
            # Add emoji based on category
            emoji = {
                "hyperparameters" : "🎛️",
                "environment"     : "🌍",
                "swarm"           : "🐦‍⬛",
                "policy"          : "🧠",
                "monitoring"      : "📊",
                "visualization"   : "📈",
            }.get(category, "⚙️")
            
            display_name = item.split('.')[-1] if '.' in item else item
            choices.append(
                questionary.Choice(f"{emoji} {display_name}", value=item)
            )
    
    choices.append(questionary.Separator("─" * 40))
    choices.append(questionary.Choice("↩️  Back to main menu", value=None))
    
    return questionary.select(
        "Select configuration to explore:",
        choices = choices,
        style   = thermal_style,
    ).ask()


def edit_config_value(
    field_name  : str,
    field_type  : str,
    current_val : any,
    description : str | None = None
) -> any:
    """
    Edit a configuration value based on its type.
    
    Args:
        field_name  : Name of the field being edited
        field_type  : Type of the field (str, int, float, bool)
        current_val : Current value
        description : Optional field description
        
    Returns:
        New value or current value if unchanged
    """
    console.print()
    
    # Show field info
    info_panel = Panel(
        f"[bold bright_cyan]{field_name}[/bold bright_cyan]\n"
        f"[muted]Type: {field_type}[/muted]\n"
        f"[muted]Current: [bright_white]{current_val}[/bright_white][/muted]" +
        (f"\n[italic]{description}[/italic]" if description else ""),
        border_style = "bright_blue",
        padding      = (1, 2),
    )
    console.print(info_panel)
    console.print()
    
    # Handle different types
    if field_type == "bool":
        new_val = questionary.confirm(
            f"Set {field_name} to:",
            default = current_val,
            style   = thermal_style,
        ).ask()
    elif field_type in ["int", "float"]:
        validator = lambda x: x.replace('.', '').replace('-', '').isdigit() if field_type == "float" else x.replace('-', '').isdigit()
        new_val = questionary.text(
            f"Enter new value for {field_name}:",
            default     = str(current_val),
            style       = thermal_style,
            validate    = validator,
        ).ask()
        new_val = float(new_val) if field_type == "float" else int(new_val)
    else:  # string
        new_val = questionary.text(
            f"Enter new value for {field_name}:",
            default = str(current_val) if current_val is not None else "",
            style   = thermal_style,
        ).ask()
    
    if new_val != current_val:
        print_message(f"Updated {field_name}: [bright_cyan]{current_val}[/bright_cyan] → [bright_green]{new_val}[/bright_green]", "success")
    else:
        print_message("Value unchanged", "info")
    
    return new_val


def prompt_for_field_value(
    field_name : str,
    field_type : str,
    current    : str
) -> any:
    """
    Prompt for a field value with type-specific validation.
    
    This is an alias for edit_config_value to match the explorer's expectations.
    
    Args:
        field_name : Name of the field
        field_type : Type string for the field
        current    : Current value as string
        
    Returns:
        New value with proper type or None if cancelled
    """
    # Convert current string to appropriate type
    if field_type == "bool":
        current_val = current.lower() == "true"
    elif field_type == "int":
        current_val = int(current) if current.isdigit() else 0
    elif field_type == "float":
        try:
            current_val = float(current)
        except ValueError:
            current_val = 0.0
    else:
        current_val = current
    
    return edit_config_value(field_name, field_type, current_val)


def select_config_component(
    title   : str,
    options : list[tuple[str, str]]
) -> str | None:
    """
    Select a configuration component from a list of options.
    
    Args:
        title   : Title for the selection prompt
        options : List of (name, description) tuples
        
    Returns:
        Selected component name or None if cancelled
    """
    print_section(title, "config")
    
    # Build choices with descriptions
    choices = []
    for name, desc in options:
        # Format the choice nicely
        choice_text = f"[bright_cyan]{name}[/bright_cyan]"
        if desc:
            # Truncate long descriptions
            if len(desc) > 50:
                desc = desc[:47] + "..."
            choice_text += f" - [muted]{desc}[/muted]"
        
        choices.append(questionary.Choice(choice_text, value=name))
    
    choices.append(questionary.Separator("─" * 40))
    choices.append(questionary.Choice("↩️  Back", value=None))
    
    selected = questionary.select(
        "Select component to explore:",
        choices = choices,
        style   = thermal_style,
    ).ask()
    
    if selected:
        print_message(f"Selected: [bright_cyan]{selected}[/bright_cyan]", "success")
    
    return selected
