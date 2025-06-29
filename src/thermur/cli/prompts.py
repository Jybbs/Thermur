"""
Orchestrates the CLI's interactive dialogues using the questionary library.

This module is responsible for the "conversation" flow of the application. It
uses the ThermurUI class to render complex components and CLIConstants for all
static text and configuration, but it defines the logic for asking questions,
gathering input, and confirming actions with the user.
"""
import questionary

from rich.align  import Align
from rich.panel  import Panel
from rich.prompt import Confirm

from .constants import CLIConstants
from .ui        import ThermurUI

ui            = ThermurUI()
thermal_style = questionary.Style(CLIConstants.Messages.QUESTIONARY_STYLE)


def select_configuration_preset() -> str | None:
    """
    Prompts the user to select a high-level configuration preset.
    
    This function first displays a descriptive table of available presets,
    then presents an interactive list. This initial choice allows users to
    quickly start with sensible defaults for different use cases without
    needing to configure every detail manually.

    Returns:
        The string name of the selected preset (e.g., "standard"), or None if
        the user explicitly chooses the "custom" configuration option.
    """
    ui.print_section("Configuration Presets", style="accent")
    
    table = ui.create_aligned_table(
        title   = CLIConstants.Presets.TABLE_TITLE,
        columns = CLIConstants.Presets.TABLE_COLUMNS,
    )
    
    preset_configs = CLIConstants.Presets.CONFIGS
    for name, config in preset_configs.items():
        if name == "custom":
            # The "custom" option is styled differently to indicate it's a
            # special case that bypasses the standard presets.
            row_style = f"[{CLIConstants.UI.MUTED_STYLE}]"
            table.add_row(
                f"{row_style}{config['name']}[/]",
                f"{row_style}{config['desc']}[/]",
                f"{row_style}{config['best_for']}[/]",
            )
        else:
            table.add_row(config['name'], config['desc'], config['best_for'])
    
    ui.console.print(table)
    ui.console.print()
    
    # Build choices for the interactive list from the same constants
    choices = [
        questionary.Choice(config['prompt'], value=config['name'])
        for name, config in preset_configs.items() if name != "custom"
    ]
    choices.extend([
        questionary.Separator(),
        questionary.Choice(
            title = preset_configs['custom']['prompt'], 
            value = None  # A value of None signifies custom configuration
        ),
    ])
    
    chosen_preset = questionary.select(
        "Which configuration preset would you like to use?",
        choices = choices,
        style   = thermal_style,
    ).ask()
    
    if chosen_preset:
        ui.print_message(
            f"Selected preset: [{CLIConstants.UI.CYAN_STYLE}]{chosen_preset}[/]",
            msg_type="success"
        )
    else:
        ui.print_message(
            "Custom configuration selected - full control mode", 
            msg_type="info"
        )
    
    return chosen_preset


def ask_wandb_project_name() -> str:
    """
    Guides the user in setting a Weights & Biases project name for tracking.

    This prompt explains the purpose of wandb and provides relevant examples
    to help the user choose a suitable project name. It falls back to a
    sensible default if no input is given.

    Returns:
        The final project name for wandb tracking.
    """
    ui.console.print()
    ui.print_message("Configure experiment tracking", "swarm")
    ui.console.print(
        f"[{CLIConstants.UI.MUTED_STYLE}]"
        "wandb will track metrics, logs, and model checkpoints"
        "[/]"
    )
    ui.console.print()
    
    examples = ui.create_examples_panel(
        items = CLIConstants.Wandb.EXAMPLE_PROJECTS,
        title = "Examples"
    )
    ui.console.print(examples)
    ui.console.print()
    
    project_name = questionary.text(
        "Enter wandb project name:",
        default     = CLIConstants.Wandb.DEFAULT_PROJECT,
        style       = thermal_style,
        instruction = "(press Enter for default)",
    ).ask()
    
    ui.print_message(f"Project name: [{CLIConstants.UI.CYAN_STYLE}]{project_name}[/]", "success")
    return project_name


def ask_for_config_overrides() -> list[str]:
    """
    Asks the user if they wish to provide advanced configuration overrides.

    If confirmed, this function enters a loop to collect multiple Hydra-style
    override strings (e.g., 'hyperparameters.lr=0.001'). It displays syntax
    examples to guide the user on the correct format.

    Returns:
        A list of configuration override strings, which may be empty.
    """
    ui.console.print()
    ui.print_section("Advanced Configuration", "config")
    
    syntax_panel = ui.create_syntax_panel(
        code  = CLIConstants.Commands.OVERRIDE_SYNTAX_HELP,
        title = CLIConstants.Commands.OVERRIDE_SYNTAX_TITLE,
    )
    ui.console.print(syntax_panel)
    ui.console.print()
    
    add_overrides = questionary.confirm(
        "Would you like to add configuration overrides?",
        default = False,
        style   = thermal_style,
    ).ask()
    
    if not add_overrides:
        return []

    ui.console.print()
    ui.console.print(
        f"[{CLIConstants.UI.MUTED_STYLE}]"
        "Enter overrides one at a time (empty line to finish):"
        "[/]"
    )
    
    overrides = []
    while True:
        override = questionary.text(
            "Override:",
            style       = thermal_style,
            instruction = "(e.g., hyperparameters.lr=0.001)",
        ).ask()
        
        if not override:
            break
        
        overrides.append(override)
        success_style = CLIConstants.Theme.STYLES['success']
        cyan_style    = CLIConstants.UI.CYAN_STYLE
        ui.console.print(f"  [{success_style}]✓[/] Added: [{cyan_style}]{override}[/]")
    
    if overrides:
        ui.console.print()
        ui.print_message(f"Added {len(overrides)} configuration override(s)", "success")
    
    return overrides


def confirm_system_override(issues: list[str]) -> bool:
    """
    Displays detected system issues and asks the user for confirmation to proceed.
    
    This function acts as a safety check, ensuring the user is aware of
    potential configuration or environment problems before continuing with a
    potentially long-running or unstable process.

    Args:
        issues: A list of string descriptions of the issues found.
        
    Returns:
        True if the user confirms they want to proceed, False otherwise.
    """
    ui.console.print()
    
    warning_panel = ui.create_warning_panel(
        title  = "⚠️  Configuration Issues Detected",
        issues = issues
    )
    ui.console.print(warning_panel)
    ui.console.print()
    
    warning_style = CLIConstants.Theme.STYLES['warning']
    return Confirm.ask(
        f"[{warning_style}]Do you want to proceed anyway?[/]",
        console = ui.console,
        default = False,
    )


def show_training_summary(config: dict) -> bool:
    """
    Presents a final summary of all chosen configurations for user confirmation.
    
    This is the last step before initiating a long-running process. It gives
    the user a final chance to review their choices (preset, wandb project,
    overrides, etc.) and either confirm or cancel the operation.

    Args:
        config: A dictionary containing the final configuration settings.
        
    Returns:
        True if the user confirms to start training, False otherwise.
    """
    ui.console.print()
    ui.print_section("Training Configuration Summary", "thermal")
    
    table = ui.create_aligned_table(
        title     = "",
        columns   = [("Setting", "bright_cyan", 20, "left"), 
                     ("Value", "bright_white", 40, "left")],
        show_edge = False,
        box       = None,
    )
    
    gpu_status   = "🎮 GPU Acceleration" if config.get("gpu_available") else "💻 CPU Mode"
    num_overrides = config.get('overrides', 0)
    
    summary_data = [
        ("Configuration", f"[{CLIConstants.Theme.STYLES['warning']}]{config.get('preset')}[/]"),
        ("wandb Project", f"[{CLIConstants.Theme.STYLES['swarm']}]{config.get('wandb_project')}[/]"),
        ("Overrides", f"[{CLIConstants.Theme.STYLES['drone']}]{num_overrides} custom settings[/]"),
        ("Hardware", gpu_status),
    ]
    for key, value in summary_data:
        table.add_row(key, value)
    
    ui.console.print(Align.center(table))
    ui.console.print()
    
    ready_panel = ui.create_ready_panel(
        title    = "✅ Ready to Train!",
        subtitle = "Your thermal swarm is configured and ready to learn"
    )
    ui.console.print(ready_panel)
    ui.console.print()
    
    success_style = CLIConstants.Theme.STYLES['success']
    return Confirm.ask(
        f"[{success_style}]Start training with this configuration?[/]",
        console = ui.console,
        default = True,
    )


def select_config_to_edit(configs: list[str]) -> str | None:
    """
    Renders an interactive list for exploring Hydra configuration groups.

    This function categorizes a flat list of configuration keys (e.g.,
    'hyperparameters.lr') into navigable groups. It uses emojis to visually
    distinguish between categories.

    Args:
        configs: A flat list of available configuration names.
        
    Returns:
        The string name of the selected configuration, or None if the user
        chooses to go back.
    """
    if not configs:
        ui.print_message("No configurations available", "warning")
        return None
    
    # Group configuration keys by their parent name (e.g., 'hyperparameters')
    categories = {}
    for cfg in configs:
        category = cfg.split('.')[0] if '.' in cfg else "main"
        if category not in categories:
            categories[category] = []
        categories[category].append(cfg)
    
    # Build the choice list for questionary with separators and emojis
    choices = []
    emojis  = CLIConstants.UI.CATEGORY_EMOJIS
    for category, items in sorted(categories.items()):
        if category != "main":
            choices.append(questionary.Separator(f"── {category.title()} ──"))
        
        for item in sorted(items):
            emoji = emojis.get(category, emojis["default"])
            display_name = item.split('.')[-1] if '.' in item else item
            choices.append(questionary.Choice(f"{emoji} {display_name}", value=item))
    
    choices.extend([
        questionary.Separator(),
        questionary.Choice("↩️  Back to main menu", value=None)
    ])
    
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
    Prompts the user to edit a single configuration value, handling basic types.
    
    This function displays the context of the field being edited (its name,
    type, current value, and description) and provides a type-aware prompt
    for the new value.

    Args:
        field_name: The name of the field being edited (e.g., "learning_rate").
        field_type: The type of the field as a string ('str', 'int', 'float', 'bool').
        current_val: The current value of the field.
        description: An optional description of the field's purpose.
        
    Returns:
        The new value, which may be the same as the current value if unchanged.
        The return type will match the requested `field_type`.
    """
    ui.console.print()
    
    # Build the content for the informational panel
    content = (f"[{CLIConstants.UI.TITLE_TEXT_STYLE}]{field_name}[/]\n"
               f"[{CLIConstants.UI.MUTED_STYLE}]Type: {field_type}[/]\n"
               f"[{CLIConstants.UI.MUTED_STYLE}]Current: "
               f"[{CLIConstants.UI.WHITE_STYLE}]{current_val}[/][/]")
    
    if description:
        content += f"\n[{CLIConstants.UI.SUBTITLE_TEXT_STYLE}]{description}[/]"
        
    info_panel = Panel(
        content,
        border_style = CLIConstants.UI.PANEL_BORDER_STYLE,
        padding      = (1, 2),
    )
    ui.console.print(info_panel)
    ui.console.print()
    
    # Provide a type-specific prompt
    if field_type == "bool":
        new_val = questionary.confirm(
            f"Set {field_name} to:", default=bool(current_val), style=thermal_style
        ).ask()
    else:
        new_val_str = questionary.text(
            f"Enter new value for {field_name}:",
            default=str(current_val) if current_val is not None else "",
            style=thermal_style,
        ).ask()
        
        # Attempt to cast the new string value to the correct type
        try:
            if new_val_str is None: # User cancelled
                return current_val
            if field_type == "int":
                new_val = int(new_val_str)
            elif field_type == "float":
                new_val = float(new_val_str)
            else: # string
                new_val = new_val_str
                
        except (ValueError, TypeError):
            ui.print_message(f"Invalid {field_type} value. Keeping original.", "error")
            return current_val

    if new_val != current_val:
        success_style = CLIConstants.Theme.STYLES['success']
        ui.print_message(f"Updated {field_name}: "
                         f"[{CLIConstants.UI.CYAN_STYLE}]{current_val}[/] → "
                         f"[{success_style}]{new_val}[/]",
                         "success")
    else:
        ui.print_message("Value unchanged", "info")
    
    return new_val


def prompt_for_field_value(
    field_name : str,
    field_type : str,
    current    : str
) -> any:
    """
    Prompts for a field value with type-specific validation (alias).
    
    This function serves as a type-converting wrapper around `edit_config_value`.
    It takes a string representation of the current value (as provided by the
    configuration explorer) and casts it to its proper Python type before
    passing it to the editing prompt.

    Args:
        field_name: The name of the configuration field.
        field_type: The type of the field as a string ('str', 'int', etc.).
        current: The current value of the field, always provided as a string.
        
    Returns:
        The new value with the proper type, or the original value if cancelled.
    """
    try:
        if field_type == "bool":
            current_val = current.lower() == "true"
        elif field_type == "int":
            current_val = int(current) if current else 0
        elif field_type == "float":
            current_val = float(current) if current else 0.0
        else:
            current_val = current
    except (ValueError, TypeError):
        current_val = current  # Fallback for complex types or parse errors
    
    return edit_config_value(field_name, field_type, current_val)


def select_config_component(
    title   : str,
    options : list[tuple[str, str]]
) -> str | None:
    """
    Presents a navigable list of components or options to the user.
    
    This function is a generic selector that takes a list of name/description
    pairs and formats them into a clean, readable, and interactive list for
    the user to choose from. It also handles truncating long descriptions
    to maintain a tidy layout.

    Args:
        title: The title to display above the list of options.
        options: A list of (name, description) tuples to be presented.
        
    Returns:
        The string `name` of the selected component, or None if the user goes back.
    """
    ui.print_section(title, "config")
    
    choices = []
    for name, desc in options:
        choice_text = f"[{CLIConstants.UI.CYAN_STYLE}]{name}[/]"
        if desc:
            # Truncate long descriptions to keep the prompt clean
            desc_limit = 50
            if len(desc) > desc_limit:
                desc = desc[:desc_limit - 3] + "..."
            choice_text += f" - [{CLIConstants.UI.MUTED_STYLE}]{desc}[/]"
        
        choices.append(questionary.Choice(choice_text, value=name))
    
    choices.extend([
        questionary.Separator(),
        questionary.Choice("↩️  Back", value=None)
    ])
    
    selected = questionary.select(
        "Select component to explore:",
        choices = choices,
        style   = thermal_style,
    ).ask()
    
    if selected:
        ui.print_message(f"Selected: [{CLIConstants.UI.CYAN_STYLE}]{selected}[/]", "success")
    
    return selected