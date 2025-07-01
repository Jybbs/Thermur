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
from typing      import Any, TypedDict

import questionary


class TrainingConfig(TypedDict, total=False):
    """Type definition for training configuration summary."""
    preset: str
    wandb_project: str
    overrides: int
    gpu_available: bool


class FieldInfo(TypedDict):
    """Type definition for field information."""
    name: str
    field_type: str
    current_val: Any
    desc: str


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
        ui: Any,  # Should be ThermurUI type when available
        prompts: DictConfig,
        messages: DictConfig
    ):
        """
        Initializes the prompt orchestrator.

        Args:
            ui       : An initialized `ThermurUI` object for rendering components.
            prompts  : Prompts configuration containing prompts-related settings.
            messages : Messages configuration containing message templates.
        """
        self.ui            = ui
        self.prompts       = prompts
        self.messages      = messages
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
            title   = self.prompts.presets_table_title,
            columns = self.prompts.presets_table_columns,
        )

        preset_configs = self.prompts.preset_configs
        for name, config in preset_configs.items():
            if name == "custom":
                row_style = f"[{self.ui.ui.muted_style}]"
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
                f"Selected preset: [{self.ui.ui.cyan_style}]{chosen_preset}[/]",
                msg_type="success"
            )
        else:
            self.ui.print_message(
                "Custom configuration selected - full control mode",
                msg_type="info"
            )

        return chosen_preset

    def ask_wandb_project_name(self) -> str:
        """
        Guides the user in setting a Weights & Biases project name for tracking.

        This prompt explains the purpose of wandb and provides relevant examples
        to help the user choose a suitable project name. It falls back to a
        sensible default if no input is given.

        Returns:
            The final project name for wandb tracking.
        """
        self.ui.console.print()
        self.ui.print_message("Configure experiment tracking", "swarm")
        self.ui.console.print(
            f"[{self.ui.ui.muted_style}]"
            "wandb will track metrics, logs, and model checkpoints"
            "[/]"
        )
        self.ui.console.print()

        examples = self.ui.create_examples_panel(
            items = self.prompts.wandb_example_projects,
            title = "Examples"
        )
        self.ui.console.print(examples)
        self.ui.console.print()

        project_name = questionary.text(
            "Enter wandb project name:",
            default     = self.prompts.wandb_default_project,
            style       = self.thermal_style,
            instruction = "(press Enter for default)",
        ).ask()

        self.ui.print_message(
            f"Project name: [{self.ui.ui.cyan_style}]{project_name}[/]", "success"
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
            code  = self.prompts.override_syntax_help,
            title = self.prompts.override_syntax_title,
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
            f"[{self.ui.ui.muted_style}]"
            "Enter overrides one at a time (empty line to finish):"
            "[/]"
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
            cyan_style    = self.ui.ui.cyan_style
            self.ui.console.print(f"  [{success_style}]✓[/] Added: [{cyan_style}]{override}[/]")

        if overrides:
            self.ui.console.print()
            self.ui.print_message(f"Added {len(overrides)} configuration override(s)", "success")

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
            field_to_edit = self.ui.console.input("[bold]Field to edit: [/bold]").strip()

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
                override_str = f"{prefix}.{name}={new_value}" if prefix else f"{name}={new_value}"
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

    def show_training_summary(self, config: TrainingConfig) -> bool:
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

        gpu_status   = "🎮 GPU Acceleration" if config.get("gpu_available") else "💻 CPU Mode"
        num_overrides = config.get('overrides', 0)

        summary_data = [
            ("Configuration", f"[{self.ui.theme.styles['warning']}]{config.get('preset')}[/]"),
            ("wandb Project", f"[{self.ui.theme.styles['swarm']}]{config.get('wandb_project')}[/]"),
            ("Overrides", f"[{self.ui.theme.styles['drone']}]{num_overrides} custom settings[/]"),
            ("Hardware", gpu_status),
        ]
        for key, value in summary_data:
            table.add_row(key, value)

        self.ui.console.print(Align.center(table))
        self.ui.console.print()

        ready_panel = self.ui.create_ready_panel(
            title    = "✅ Ready to Train!",
            subtitle = "Your thermal swarm is configured and ready to learn"
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
        emojis  = self.ui.ui.category_emojis
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

        content = (f"[{self.ui.ui.title_text_style}]{field_name}[/]\n"
                   f"[{self.ui.ui.muted_style}]Type: {field_type}[/]\n"
                   f"[{self.ui.ui.muted_style}]Current: "
                   f"[{self.ui.ui.white_style}]{current_val}[/][/]")

        if description:
            content += f"\n[{self.ui.ui.subtitle_text_style}]{description}[/]"

        info_panel = Panel(
            content,
            border_style = self.ui.ui.panel_border_style,
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
                if field_type == "int":
                    new_val = int(new_val_str)
                elif field_type == "float":
                    new_val = float(new_val_str)
                else:
                    new_val = new_val_str
            except (ValueError, TypeError):
                self.ui.print_message(f"Invalid {field_type} value. Keeping original.", "error")
                return current_val

        if new_val != current_val:
            success_style = self.ui.theme.styles['success']
            self.ui.print_message(f"Updated {field_name}: "
                             f"[{self.ui.ui.cyan_style}]{current_val}[/] → "
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
        self.ui.print_section(title, "config")

        choices = []
        for name, desc in options:
            choice_text = f"[{self.ui.ui.cyan_style}]{name}[/]"

            if desc:
                desc_limit = 50
                if len(desc) > desc_limit:
                    desc = desc[:desc_limit - 3] + "..."
                choice_text += f" - [{self.ui.ui.muted_style}]{desc}[/]"

            choices.append(questionary.Choice(choice_text, value=name))

        choices.extend([
            questionary.Separator(),
            questionary.Choice("↩️  Back", value=None)
        ])

        selected = questionary.select(
            "Select component to explore:",
            choices = choices,
            style   = self.thermal_style,
        ).ask()

        if selected:
            self.ui.print_message(
                f"Selected: [{self.ui.ui.cyan_style}]{selected}[/]", "success"
            )

        return selected