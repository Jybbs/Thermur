"""
Orchestrates the CLI's interactive dialogues using the questionary library.

This module manages the interactive prompts for the training command, including
preset selection, wandb configuration, override collection, and training 
confirmation. It uses the ThermurUI class to render complex components and 
DictConfig objects for all static text and configuration.
"""
from omegaconf   import DictConfig
from rich.align  import Align
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
        cli      : DictConfig,
        messages : DictConfig,
        presets  : DictConfig,
        prompts  : DictConfig,
        ui       : Any
    ):
        """
        Initializes the prompt orchestrator.

        Args:
            cli      : CLI configuration containing command-related settings.
            messages : Messages configuration containing message templates.
            presets  : Presets configuration containing training preset definitions.
            prompts  : Prompts configuration containing prompts-related settings.
            ui       : An initialized `ThermurUI` object for rendering components.
        """
        self.cli      = cli
        self.messages = messages
        self.presets  = presets
        self.prompts  = prompts
        self.ui       = ui
        self.thermal_style = questionary.Style.from_dict(
            styles = dict(self.prompts.questionary_style)
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
        self.ui.print_minor_section("Configuration Presets")

        table = self.ui.create_aligned_table(
            columns = self.prompts.presets_table_columns,
            title   = "Available Presets"
        )

        preset_cfgs = self.presets.presets
        for name, config in preset_cfgs.items():
            display_name = f"{config['emoji']} {config['name']}"
            if name == "custom":
                row_style = f"[grey70]"
                table.add_row(
                    f"{row_style}{display_name}[/]",
                    f"{row_style}{config['desc']}[/]",
                    f"{row_style}{config['best_for']}[/]",
                )
            else:
                table.add_row(display_name, config['desc'], config['best_for'])

        self.ui.console.print(table)
        self.ui.console.print()

        choices = [
            questionary.Choice(
                title = preset_cfgs[name]['emoji'], 
                value = name
            )
            for name in list(self.config.presets.presets.keys())
        ]
        choices.extend([
            questionary.Separator(),
            questionary.Choice(
                title = preset_cfgs['custom']['emoji'],
                value = None
            ),
        ])

        chosen_emoji = questionary.select(
            choices = choices,
            message = "Which configuration preset would you like to use?",
            style   = self.thermal_style
        ).ask()

        # Map emoji back to preset name
        if chosen_emoji:
            # Find which preset has this emoji
            chosen_preset = None
            for name, config in preset_cfgs.items():
                if config['emoji'] == chosen_emoji:
                    chosen_preset = name
                    break
            
            self.ui.print_message(
                message  = f"Selected preset: [bright_cyan]{chosen_emoji}[/bright_cyan]",
                msg_type = "success"
            )
        else:
            chosen_preset = None
            self.ui.print_message(
                message  = "Custom configuration selected - full control mode",
                msg_type = "info"
            )

        return chosen_preset

    def ask_wandb_project_name(self, default_project: str = "thermur") -> str:
        """
        Guides the user in setting a Weights & Biases project name for tracking.

        This prompt explains the purpose of wandb before allowing the user
        to enter their project name.

        Args:
            default_project: The default project name to suggest.

        Returns:
            The final project name for wandb tracking.
        """
        self.ui.console.print()
        self.ui.print_message(
            message  = "Configure experiment tracking",
            msg_type = "flock"
        )
        self.ui.console.print(
            f"[grey70]"
            "wandb will track metrics, logs, and model checkpoints"
            "[/grey70]"
        )
        self.ui.console.print()


        project_name = questionary.text(
            default     = default_project,
            instruction = "(press Enter for default)",
            message     = "Enter wandb project name:",
            style       = self.thermal_style
        ).ask()

        self.ui.print_message(
            message  = f"Project name: [bright_cyan]{project_name}[/bright_cyan]",
            msg_type = "success"
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
        self.ui.print_minor_section("Advanced Configuration")

        syntax_panel = self.ui.create_syntax_panel(
            code  = self.cli.override_syntax_help,
            title = "Configuration Override Syntax",
        )
        self.ui.console.print(syntax_panel)
        self.ui.console.print()

        add_overrides = questionary.confirm(
            default = False,
            message = "Would you like to add configuration overrides?",
            style   = self.thermal_style
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
                instruction = "(e.g., hyperparameters.lr=0.001)",
                message     = "Override:",
                style       = self.thermal_style
            ).ask()

            if not override:
                break

            overrides.append(override)
            success_style = self.ui.display.styles['success']
            self.ui.console.print(
                f"  [{success_style}]✓[/] Added: [bright_cyan]{override}[/bright_cyan]"
            )

        if overrides:
            self.ui.console.print()
            self.ui.print_message(
                message  = f"Added {len(overrides)} configuration override(s)",
                msg_type = "success"
            )

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
            issues = issues,
            title  = "⚠️  Configuration Issues Detected"
        )
        self.ui.console.print(warning_panel)
        self.ui.console.print()

        warning_style = self.ui.display.styles['warning']
        return Confirm.ask(
            console = self.ui.console,
            default = False,
            prompt  = f"[{warning_style}]Do you want to proceed anyway?[/]"
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
        self.ui.print_minor_section("Training Configuration Summary")

        table = self.ui.create_aligned_table(
            box       = None,
            columns   = [
                ("Setting", "bright_cyan",  20, "left"),
                ("Value",   "bright_white", 40, "left")
            ],
            show_edge = False,
            title     = ""
        )

        num_overrides = config.get('overrides', 0)
        summary_data  = [
            (
                "Configuration",
                f"[{self.ui.display.styles['warning']}]{config.get('preset')}[/]"
            ),
            (
                "Hardware", 
                "🎮 GPU Acceleration" if config.get("gpu_available") else "💻 CPU Mode"
            ),
            (
                "Overrides", 
                f"[{self.ui.display.styles['drone']}]{num_overrides} custom settings[/]"
            ),
            (
                "wandb Project",
                f"[{self.ui.display.styles['flock']}]{config.get('wandb_project')}[/]"
            )
        ]
        for key, value in summary_data:
            table.add_row(key, value)

        self.ui.console.print(Align.center(table))
        self.ui.console.print()

        ready_panel = self.ui.create_ready_panel(
            subtitle = "Your thermal flock is configured and ready to fly.",
            title    = "✅ Ready to train!"
        )
        self.ui.console.print(ready_panel)
        self.ui.console.print()

        return Confirm.ask(
            "[bold bright_green]Start training with this configuration?[/]",
            console = self.ui.console,
            default = True
        )
