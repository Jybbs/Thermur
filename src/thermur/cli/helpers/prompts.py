"""
Orchestrates the CLI's interactive dialogues using the questionary library.

This module manages the interactive prompts for the training command, including
preset selection, wandb configuration, override collection, and training 
confirmation. It uses the ThermurUI class to render complex components and 
DictConfig objects for all static text and configuration.
"""
from omegaconf import DictConfig
from typing    import Any, Callable

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
        self.cfg     = cfg
        self.thermal = questionary.Style.from_dict(cfg.display.questionary_style)
        self.ui      = ui

    def ask_for_overrides(self) -> list[str]:
        """
        Asks the user if they wish to provide advanced configuration overrides.

        If confirmed, this function enters a loop to collect multiple Hydra-style
        override strings (e.g., 'optimizer.learning_rate=0.001'). It displays syntax
        examples to guide the user on the correct format.

        Returns:
            A list of configuration override strings, which may be empty.
        """
        self.ui.console.print()
        self.ui.print_section("Advanced Configuration", minor=True)

        syntax_panel = self.ui.create_syntax_panel(
            code  = self.cfg.display.override_examples,
            title = "Configuration Override Syntax",
        )
        self.ui.console.print(syntax_panel)
        self.ui.console.print()

        add_overrides = questionary.confirm(
            default = False,
            message = "Would you like to add configuration overrides?",
            style   = self.thermal
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
            instruction = "(e.g., optimizer.learning_rate=0.001)",
            message     = "Override:",
            style       = self.thermal
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
            style   = self.thermal
        ).ask()
    
    def confirm_deletion(
        self, 
        count  : int,
        items  : str = "items",
        keep   : int = 0,
        force  : bool = False
    ) -> bool:
        """
        Confirm deletion of items with warning panel.
        
        Displays a warning panel with deletion details and prompts for confirmation.
        Useful for any destructive operation that needs user confirmation.
        
        Args:
            count  : Number of items to delete
            items  : Plural name of items (e.g., "runs", "files")
            keep   : Number of items being kept (0 if deleting all)
            force  : Skip confirmation if True
            
        Returns:
            True if user confirms deletion, False otherwise
        """
        if force:
            return True
            
        # Build warning messages
        issues = [
            f"This will permanently delete {count} {items}",
            "This action cannot be undone"
        ]
        
        if keep > 0:
            issues.append(f"Keeping only the {keep} most recent {items}")
        else:
            issues.append(f"Use --keep N to preserve N recent {items}")
            
        # Display warning panel
        warning_panel = self.ui.create_warning_panel(
            issues = issues,
            title  = "⚠️  Confirm Deletion"
        )
        self.ui.display_panel(warning_panel)
        
        # Prompt for confirmation
        return self.confirm(f"Delete {count} {items}?")
    
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
    
    def paginate(
        self,
        items            : list[Any],
        render_page      : Callable,
        allow_row_select : bool = True,
        page_size        : int  = 10
    ) -> Any | None:
        """
        Generic pagination handler for any list of items.
        
        Provides a reusable pagination interface that delegates page rendering
        to a callback function. Handles navigation, selection, and clearing.
        Displays navigation options and handles user input.
        
        Args:
            items            : List of items to paginate
            render_page      : Callback function to render each page
            allow_row_select : Whether to allow row selection
            page_size        : Number of items per page
            
        Returns:
            Selected item if allow_row_select=True and item selected, None otherwise
        """
        if not items:
            return None
            
        page = 0
        total_pages = -(-len(items) // page_size)
        
        first_render = True
        
        while True:
            start      = page * page_size
            page_items = items[start:start + page_size]
            
            if not first_render:
                lines_to_clear = len(page_items) + 20
                self.ui.console.file.write(f'\033[{lines_to_clear}A')
                
                for _ in range(lines_to_clear):
                    self.ui.console.file.write('\033[K\n')
                
                self.ui.console.file.write(f'\033[{lines_to_clear}A')
                self.ui.console.file.flush()
            
            render_page(page_items, page + 1, total_pages)
            nav_options = []
            
            if allow_row_select and len(page_items) > 0:
                nav_options.append(("[0-9] Select", ""))
            
            if page > 0:
                nav_options.append(("[P]revious", "p"))
            if page < total_pages - 1:
                nav_options.append(("[N]ext", "n"))
                
            nav_options.append(("[Q]uit", "q"))
            
            nav_display = " | ".join(
                f"[bold cyan]{opt[0]}[/]" for opt in nav_options
            )
            self.ui.console.print(nav_display + "\n")
            
            valid_choices = ["p", "n", "q", ""]
            if allow_row_select and len(page_items) > 0:
                valid_choices.extend(map(str, range(len(page_items))))
            
            try:
                choice = questionary.text(
                    "Select",
                    style    = self.thermal,
                    validate = lambda x: x.strip().lower() in valid_choices
                ).ask()
                
                if not choice:
                    return None
            except Exception:
                self.ui.console.print("[dim]Enter choice: [/dim]", end="")
                choice = input().strip().lower()
                if choice not in valid_choices:
                    continue
            
            match choice.strip().lower():
                case "q" | "":
                    return None
                case "p":
                    if page > 0:
                        page = max(0, page - 1)
                        first_render = False
                case "n":
                    if page < total_pages - 1:
                        page = min(total_pages - 1, page + 1)
                        first_render = False
                case n if n.isdigit() and allow_row_select:
                    return page_items[int(n)]
    
    def select_file_from_pages(
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
        def render_file_page(
            page_files  : list[dict], 
            page_num    : int, 
            total_pages : int
        ):
            """
            Render a page of files with download status table.
            """
            self.ui.display_download_table(
                available_files = page_files,
                file_status     = file_status,
                title           = f"{title_prefix} (Page {page_num}/{total_pages})"
            )
            self.ui.console.print()
        
        return self.paginate(
            allow_row_select = True,
            items            = available_files,
            page_size        = page_size,
            render_page      = render_file_page
        )
    
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
            style   = self.thermal,
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
