"""
Orchestrates the CLI's interactive dialogues using the questionary library.

This module manages the interactive prompts for the training command, including
preset selection, wandb configuration, override collection, and training 
confirmation. It uses the ThermurUI class to render complex components and 
DictConfig objects for all static text and configuration.
"""
from itertools import islice
from omegaconf import DictConfig
from typing    import Any

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
        cfg : DictConfig,
        ui  : Any
    ):
        """
        Initializes the prompt orchestrator.

        Args:
            cfg : The full configuration object containing all subsections.
            ui  : An initialized `ThermurUI` object for rendering components.
        """
        self.cfg      = cfg
        self.cli      = cfg.cli
        self.messages = cfg.messages
        self.presets  = cfg.presets
        self.prompts  = cfg.prompts
        self.ui       = ui
        self.thermal_style = questionary.Style.from_dict(
            dict(self.prompts.questionary_style)
        )

    def ask_for_overrides(self) -> list[str]:
        """
        Asks the user if they wish to provide advanced configuration overrides.

        If confirmed, this function enters a loop to collect multiple Hydra-style
        override strings (e.g., 'hyperparameters.lr=0.001'). It displays syntax
        examples to guide the user on the correct format.

        Returns:
            A list of configuration override strings, which may be empty.
        """
        self.ui.console.print()
        self.ui.print_section("Advanced Configuration", minor=True)

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
        while override := questionary.text(
            instruction = "(e.g., hyperparameters.lr=0.001)",
            message     = "Override:",
            style       = self.thermal_style
        ).ask():
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
    
    def confirm(
        self, 
        message : str, 
        default : bool = True
    ) -> bool:
        """
        General purpose confirmation prompt.
        
        Args:
            message : The question to ask the user
            default : Default value if user just presses enter
            
        Returns:
            True if user confirms, False otherwise
        """
        return questionary.confirm(
            default = default,
            message = message,
            style   = self.thermal_style
        ).ask()
    
    def confirm_download(self, file_info: dict) -> bool:
        """
        Prompts user to confirm file download operation.
        
        Args:
            file_info: File information dictionary with name and size
            
        Returns:
            True if user confirms download, False otherwise
        """
        size_gb = file_info['size'] / 1e9
        
        self.ui.console.print()
        self.ui.print_message(
            f"Ready to download: {file_info['name']}",
            "info"
        )
        self.ui.console.print(f"[yellow]File size: {size_gb:.1f} GB[/yellow]")
        self.ui.console.print(
            "[grey70]This download may take several hours depending on your "
            "internet connection[/grey70]"
        )
        self.ui.console.print()
        
        return self.confirm(message = "Proceed with download?")

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

        return self.confirm(
            message = "Do you want to proceed anyway?",
            default = False
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
        self.ui.print_section("Configuration Presets", minor=True)

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
            for name in self.presets.presets.keys()
            if name != 'custom'
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

        if chosen_emoji:
            emoji_to_preset = {
                config['emoji']: name 
                for name, config in preset_cfgs.items()
            }
            chosen_preset = emoji_to_preset.get(chosen_emoji)
            
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
    
    def select_file_with_pagination(
        self,
        available_files : list[dict],
        file_status     : dict[str, str],
        page_size       : int = 10,
        title_prefix    : str = "Available Files"
    ) -> dict | None:
        """
        Display files in paginated table format and allow selection.
        
        Presents a paginated view of available files with their download status,
        allowing users to navigate between pages and select files using keyboard
        commands. Uses Rich tables for display and questionary for input handling.
        
        Args:
            available_files : List of file dictionaries with 'name' and 'size' keys
            file_status     : Dict mapping filename to status
            page_size       : Number of files to display per page
            title_prefix    : Prefix for the page title display
            
        Returns:
            Selected file dictionary or None if cancelled
        """
        page, total = 0, len(available_files)
        pages = -(-total // page_size)
        
        while True:
            self.ui.console.clear()
            
            start = page * page_size
            page_files = list(islice(available_files, start, start + page_size))
            
            self.ui.display_download_table(
                available_files = page_files,
                file_status     = file_status,
                title           = f"{title_prefix} (Page {page + 1}/{pages})"
            )
            self.ui.console.print()
            self.ui.display_download_summary(available_files, file_status)
            self.ui.console.print()
            
            choices = list(map(str, range(len(page_files)))) + ["q", ""]
            nav     = ["[bold cyan]0-9[/]: Select", "[bold cyan]q[/]: Quit"]
            
            if page:
                choices.append("p")
                nav.insert(1, "[bold cyan]p[/]: Previous")
            if start + page_size < total:
                choices.append("n") 
                nav.insert(-1, "[bold cyan]n[/]: Next")
            
            self.ui.console.print("  ".join(nav) + "\n")
            
            if not (choice := questionary.text(
                "Select",
                style    = self.thermal_style,
                validate = lambda x: x.strip().lower() in choices
            ).ask()):
                return None
            
            match choice.strip().lower():
                case "q" | "": return None
                case "p": page -= 1
                case "n": page += 1  
                case n if n.isdigit():
                    return page_files[int(n)]
    
    def select_from_list(
        self,
        choices : list[tuple[str, str]],
        message : str
    ) -> str | None:
        """
        Present a list of choices for selection.
        
        Args:
            choices : List of (value, description) tuples
            message : The prompt message
            
        Returns:
            Selected value or None if cancelled
        """
        return questionary.select(
            choices = [
                questionary.Choice(desc, val) 
                for val, desc in choices
            ],
            message = message,
            style   = self.thermal_style,
            instruction = "(↑↓)"
        ).ask()
            
    def select_globus_endpoint(self, endpoints: list[dict]) -> dict | None:
        """
        Select a Globus endpoint from multiple available endpoints.
        
        When multiple local Globus endpoints are found, prompts the user
        to select which one to use for transfers.
        
        Args:
            endpoints: List of endpoint dictionaries with 'display_name' and 'id'
            
        Returns:
            Selected endpoint dict, or None if cancelled
        """
        if not endpoints:
            return None
            
        if len(endpoints) == 1:
            return endpoints[0]
            
        self.ui.print_message("Multiple local endpoints found:", "info")
        for i, e in enumerate(endpoints, start=1):
            self.ui.console.print(f"  {i}. {e['display_name']}")
            
        selected = self.select_from_list(
            choices = [(e, e['display_name']) for e in endpoints],
            message = "Select local endpoint:"
        )
        
        return selected
            
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
        self.ui.print_section("Training Configuration Summary", minor=True)

        table = self.ui.create_aligned_table(
            box       = None,
            columns   = [
                ("Setting", "bright_cyan",  20, "left"),
                ("Value",   "bright_white", 40, "left")
            ],
            show_edge = False
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

        self.ui.console.print(table, justify="center")
        self.ui.console.print()

        ready_panel = self.ui.create_ready_panel(
            subtitle = "Your thermal flock is configured and ready to fly.",
            title    = "✅ Ready to train!"
        )
        self.ui.console.print(ready_panel)
        self.ui.console.print()

        return self.confirm(
            message = "Start training with this configuration?",
            default = True
        )
