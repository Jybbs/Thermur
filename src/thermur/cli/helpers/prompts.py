"""
Orchestrates the CLI's interactive dialogues using the questionary library.

This module is responsible for the "conversation" flow of the application. It
uses the ThermurUI class to render complex components and DictConfig objects for all
static text and configuration, but it defines the logic for asking questions,
gathering input, and confirming actions with the user.
"""
from omegaconf   import DictConfig
from rich.align  import Align
from rich.panel  import Panel
from rich.prompt import Confirm
from typing      import Any

import questionary


class CLIPrompts:
    """
    Manages all interactive command-line dialogues for the Thermur CLI.

    This class encapsulates the logic for asking the user questions, presenting
    choices, and confirming actions. It relies on a `ThermurUI` instance and 
    DictConfig objects for prompts and messages, both provided during initialization,
    to render visuals and access static text. This keeps the interactive logic separate
    from both the UI rendering and the core application orchestration.
    """
    def __init__(
        self, 
        ui       : Any,
        prompts  : DictConfig,
        messages : DictConfig,
        commands : DictConfig
    ):
        """
        Initializes the prompt orchestrator.

        Args:
            ui       : An initialized `ThermurUI` object for rendering components.
            prompts  : Prompts configuration containing prompts-related settings.
            messages : Messages configuration containing message templates.
            commands : Commands configuration containing command-related settings.
        """
        self.ui            = ui
        self.prompts       = prompts
        self.messages      = messages
        self.commands      = commands
        self.thermal_style = questionary.Style.from_dict(
            dict(self.prompts.questionary_style)
        )

    def select_configuration_preset(self) -> str | None:
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
        self.ui.print_section("Configuration Presets", style="accent")

        table = self.ui.create_aligned_table(
            title   = "Available Presets",
            columns = self.prompts.presets_table_columns,
        )

        preset_configs = self.prompts.preset_configs
        for name, config in preset_configs.items():
            if name == "custom":
                row_style = f"[grey70]"
                table.add_row(
                    f"{row_style}{config['name']}[/]",
                    f"{row_style}{config['desc']}[/]",
                    f"{row_style}{config['best_for']}[/]",
                )
            else:
                table.add_row(config['name'], config['desc'], config['best_for'])

        self.ui.console.print(table)
        self.ui.console.print()

        choices = [
            questionary.Choice(config['prompt'], value=config['name'])
            for name, config in preset_configs.items() if name != "custom"
        ]
        choices.extend([
            questionary.Separator(),
            questionary.Choice(
                title = preset_configs['custom']['prompt'],
                value = None
            ),
        ])

        chosen_preset = questionary.select(
            "Which configuration preset would you like to use?",
            choices = choices,
            style   = self.thermal_style,
        ).ask()

        if chosen_preset:
            self.ui.print_message(
                f"Selected preset: [bright_cyan]{chosen_preset}[/bright_cyan]",
                msg_type="success"
            )
        else:
            self.ui.print_message(
                "Custom configuration selected - full control mode",
                msg_type="info"
            )

        return chosen_preset

    def ask_wandb_project_name(self, default_project: str = "thermur") -> str:
        """
        Guides the user in setting a Weights & Biases project name for tracking.

        This prompt explains the purpose of wandb before allowing the user
        to enter their project name.

        Args:
            default_project : The default project name to suggest.

        Returns:
            The final project name for wandb tracking.
        """
        self.ui.console.print()
        self.ui.print_message("Configure experiment tracking", "flock")
        self.ui.console.print(
            f"[grey70]"
            "wandb will track metrics, logs, and model checkpoints"
            "[/grey70]"
        )
        self.ui.console.print()


        project_name = questionary.text(
            "Enter wandb project name:",
            default     = default_project,
            style       = self.thermal_style,
            instruction = "(press Enter for default)",
        ).ask()

        self.ui.print_message(
            f"Project name: [bright_cyan]{project_name}[/bright_cyan]", "success"
        )
        return project_name

    def ask_for_config_overrides(self) -> list[str]:
        """
        Asks the user if they wish to provide advanced configuration overrides.

        If confirmed, this function enters a loop to collect multiple Hydra-style
        override strings (e.g., 'hyperparameters.lr=0.001'). It displays syntax
        examples to guide the user on the correct format.

        Returns:
            A list of configuration override strings, which may be empty.
        """
        self.ui.console.print()
        self.ui.print_section("Advanced Configuration", "config")

        syntax_panel = self.ui.create_syntax_panel(
            code  = self.commands.override_syntax_help,
            title = "Configuration Override Syntax",
        )
        self.ui.console.print(syntax_panel)
        self.ui.console.print()

        add_overrides = questionary.confirm(
            "Would you like to add configuration overrides?",
            default = False,
            style   = self.thermal_style,
        ).ask()

        if not add_overrides:
            return []

        self.ui.console.print()
        self.ui.console.print(
            f"[grey70]"
            "Enter overrides one at a time (empty line to finish):"
            "[/grey70]"
        )

        overrides = []
        while True:
            override = questionary.text(
                "Override:",
                style       = self.thermal_style,
                instruction = "(e.g., hyperparameters.lr=0.001)",
            ).ask()

            if not override:
                break

            overrides.append(override)
            success_style = self.ui.theme.styles['success']
            self.ui.console.print(
                f"  [{success_style}]✓[/] Added: [bright_cyan]{override}[/bright_cyan]"
            )

        if overrides:
            self.ui.console.print()
            self.ui.print_message(
                f"Added {len(overrides)} configuration override(s)", "success"
            )

        return overrides

    def edit_multiple_fields(
        self,
        fields      : list[tuple[str, str, Any, str]],
        prefix      : str,
        description : str,
    ) -> list[str]:
        """
        Runs an interactive loop to prompt the user to edit multiple fields.

        Args:
            fields      : A list of tuples, each containing info about a field.
            prefix      : The dot-path prefix for generating Hydra overrides.
            description : A message to display to the user before prompting.

        Returns:
            A list of generated override strings.
        """
        self.ui.print_message(description, "info")
        overrides = []

        while True:
            field_to_edit = self.ui.console.input(
                "[bold]Field to edit: [/bold]"
            ).strip()

            if not field_to_edit:
                break

            field_data = next((f for f in fields if f[0] == field_to_edit), None)

            if not field_data:
                self.ui.print_message(f"Field '{field_to_edit}' not found", "warning")
                continue

            name, field_type, current_val, desc = field_data

            new_value = self.prompt_for_field_value(
                field_name  = name,
                field_type  = field_type,
                current     = str(current_val),
                description = desc,
            )

            if new_value is not None and str(new_value) != str(current_val):
                override_str = (
                    f"{prefix}.{name}={new_value}" 
                    if prefix else f"{name}={new_value}"
                )
                overrides.append(override_str)
                self.ui.print_message(f"Added override: {override_str}", "success")

        return overrides

    def confirm_system_override(self, issues: list[str]) -> bool:
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
        self.ui.console.print()

        warning_panel = self.ui.create_warning_panel(
            title  = "⚠️  Configuration Issues Detected",
            issues = issues
        )
        self.ui.console.print(warning_panel)
        self.ui.console.print()

        warning_style = self.ui.theme.styles['warning']
        return Confirm.ask(
            f"[{warning_style}]Do you want to proceed anyway?[/]",
            console = self.ui.console,
            default = False,
        )

    def show_training_summary(self, config: dict[str, Any]) -> bool:
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
        self.ui.console.print()
        self.ui.print_section("Training Configuration Summary", "thermal")

        table = self.ui.create_aligned_table(
            title     = "",
            columns   = [("Setting", "bright_cyan", 20, "left"),
                         ("Value", "bright_white", 40, "left")],
            show_edge = False,
            box       = None,
        )

        gpu_status = (
            "🎮 GPU Acceleration" if config.get("gpu_available") 
            else "💻 CPU Mode"
        )
        num_overrides = config.get('overrides', 0)

        summary_data = [
            (
                "Configuration",
                f"[{self.ui.theme.styles['warning']}]{config.get('preset')}[/]"
            ),
            (
                "wandb Project",
                f"[{self.ui.theme.styles['flock']}]{config.get('wandb_project')}[/]"
            ),
            (
                "Overrides", 
                f"[{self.ui.theme.styles['drone']}]{num_overrides} custom settings[/]"
            ),
            (
                "Hardware", 
                gpu_status
            ),
        ]
        for key, value in summary_data:
            table.add_row(key, value)

        self.ui.console.print(Align.center(table))
        self.ui.console.print()

        ready_panel = self.ui.create_ready_panel(
            title    = "✅ Ready to Train!",
            subtitle = "Your thermal flock is configured and ready to learn"
        )
        self.ui.console.print(ready_panel)
        self.ui.console.print()

        success_style = self.ui.theme.styles['success']
        return Confirm.ask(
            f"[{success_style}]Start training with this configuration?[/]",
            console = self.ui.console,
            default = True,
        )

    def select_config_to_edit(self, configs: list[str]) -> str | None:
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
            self.ui.print_message("No configurations available", "warning")
            return None

        categories = {}
        for cfg in configs:
            category = cfg.split('.')[0] if '.' in cfg else "main"
            if category not in categories:
                categories[category] = []
            categories[category].append(cfg)

        choices = []
        emojis  = {"standard": "", "custom": "✨", "advanced": "🔧"}
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
            style   = self.thermal_style,
        ).ask()

    def edit_config_value(
        self,
        field_name  : str,
        field_type  : str,
        current_val : Any,
        description : str | None = None
    ) -> Any:
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
        self.ui.console.print()

        content = (f"[bold bright_cyan]{field_name}[/bold bright_cyan]\n"
                   f"[grey70]Type: {field_type}[/grey70]\n"
                   f"[grey70]Current: "
                   f"[white]{current_val}[/white][/grey70]")

        if description:
            content += f"\n[muted italic]{description}[/muted italic]"

        info_panel = Panel(
            content,
            border_style = "bright_blue",
            padding      = (1, 2),
        )
        self.ui.console.print(info_panel)
        self.ui.console.print()

        if field_type == "bool":
            new_val = questionary.confirm(
                f"Set {field_name} to:", default=bool(current_val), style=self.thermal_style
            ).ask()
        else:
            new_val_str = questionary.text(
                f"Enter new value for {field_name}:",
                default=str(current_val) if current_val is not None else "",
                style=self.thermal_style,
            ).ask()

            try:
                if new_val_str is None:
                    return current_val
                    
                new_val = self._convert_value(new_val_str, field_type)
            except (ValueError, TypeError) as e:
                self.ui.print_message(f"Invalid {field_type} value: {e}. Keeping original.", "error")
                return current_val

        if new_val != current_val:
            success_style = self.ui.theme.styles['success']
            self.ui.print_message(f"Updated {field_name}: "
                             f"[bright_cyan]{current_val}[/bright_cyan] → "
                             f"[{success_style}]{new_val}[/]",
                             "success")
        else:
            self.ui.print_message("Value unchanged", "info")

        return new_val

    def prompt_for_field_value(
        self,
        field_name : str,
        field_type : str,
        current    : str,
        description: str | None = None
    ) -> Any:
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
            description: An optional description of the field's purpose.

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
            current_val = current

        return self.edit_config_value(field_name, field_type, current_val, description)

    def select_config_component(
        self,
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
        choices = []
        for name, desc in options:
            # Truncate description if too long
            if len(desc) > 70:
                desc = desc[:67] + "..."
            choices.append(questionary.Choice(f"{name} - {desc}", value=name))
        
        choices.extend([
            questionary.Separator(),
            questionary.Choice("↩️  Back to previous menu", value=None)
        ])
        
        selected = questionary.select(
            title,
            choices = choices,
            style   = self.thermal_style,
        ).ask()
        
        if selected:
            self.ui.print_message(
                f"Selected: [bright_cyan]{selected}[/bright_cyan]", "success"
            )
        
        return selected
    
    # Removed select_config_with_table - no longer needed
    
    def _convert_value(self, value_str: str, field_type: str) -> Any:
        """
        Convert a string value to the appropriate type.
        
        Args:
            value_str: The string value to convert
            field_type: The target type as a string
            
        Returns:
            The converted value
        """
        if field_type == "int":
            return int(value_str)
        elif field_type == "float":
            return float(value_str)
        elif field_type == "bool":
            return value_str.lower() in ("true", "yes", "1", "on")
        elif field_type.startswith("list"):
            # Handle list types like list[str], list[int], etc.
            import json
            try:
                # Try JSON parsing first
                val = json.loads(value_str)
                if isinstance(val, list):
                    return val
            except:
                # Fall back to comma-separated values
                items = [item.strip() for item in value_str.split(",")]
                # Try to infer item type from field_type
                if "[int]" in field_type:
                    return [int(item) for item in items]
                elif "[float]" in field_type:
                    return [float(item) for item in items]
                else:
                    return items
        elif field_type.startswith("dict"):
            import json
            return json.loads(value_str)
        elif field_type.startswith("tuple"):
            import json
            val = json.loads(value_str)
            return tuple(val) if isinstance(val, list) else val
        elif field_type == "Path":
            from pathlib import Path
            return Path(value_str)
        else:
            # Default to string
            return value_str
    
    def confirm_training_start(self) -> bool:
        """
        Ask user to confirm starting training with current configuration.
        
        Returns:
            True if user confirms, False otherwise.
        """
        return questionary.confirm(
            self.messages.ready_to_train,
            default=True,
            style=self.thermal_style
        ).ask()