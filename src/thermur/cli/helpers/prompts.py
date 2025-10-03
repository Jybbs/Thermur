"""
Orchestrates the CLI's interactive dialogues using the questionary library.

This module manages the interactive prompts for the training command, including
wandb configuration, override collection, and training confirmation. It uses
the ThermurUI class to render complex components and structured configuration
objects for all static text and configuration.
"""
from __future__   import annotations
from config.types import TableColumn
from itertools    import islice
from typing       import Any, Callable, Sequence, TYPE_CHECKING

import questionary

if TYPE_CHECKING:
    from config.cli.builds import CLIConfiguration
    from .ui               import ThermurUI


class CLIPrompts:
    """
    Manages all interactive command-line dialogues for the Thermur CLI.

    This class encapsulates the logic for asking the user questions, presenting
    choices, and confirming actions. It relies on a `ThermurUI` instance and
    CLIConfiguration for prompts and messages, both provided during initialization,
    to render visuals and access static text. This keeps the interactive logic separate
    from both the UI rendering and the core application orchestration.
    """
    def __init__(
        self,
        cfg : CLIConfiguration,
        ui  : ThermurUI
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

        overrides: list[str] = []
        while override := questionary.text(
            instruction = "(e.g., training.optimizer.learning_rate=0.001)",
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
        items            : Sequence[Any],
        render_page      : Callable[[list[Any], int, int], None],
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

        page        = 0
        items_count = len(items)
        total_pages = -(-items_count // page_size)

        # Save cursor position before pagination starts
        self.ui.console.file.write('\033[s')
        self.ui.console.file.flush()

        while True:
            start      = page * page_size
            page_items = list(islice(items, start, start + page_size))

            # Restore cursor and clear to end of screen
            self.ui.console.file.write('\033[u\033[J')
            self.ui.console.file.flush()

            render_page(page_items, page + 1, total_pages)
            nav_options: list[tuple[str, str]] = []

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

            valids = ["p", "n", "q", ""] + (
                [str(i) for i in range(len(page_items))]
                if allow_row_select and page_items
                else []
            )

            validate: Callable[[str], bool] = lambda x: x.strip().lower() in valids
            try:
                choice = questionary.text(
                    "Select",
                    style    = self.thermal,
                    validate = validate
                ).ask()

                if not choice:
                    return None

            except Exception:
                self.ui.console.print("[dim]Enter choice: [/dim]", end="")
                choice = input().strip().lower()
                if choice not in valids:
                    continue

            match choice.strip().lower():
                case "q" | "":
                    return None
                case "p":
                    if page > 0:
                        page = max(0, page - 1)
                case "n":
                    if page < total_pages - 1:
                        page = min(total_pages - 1, page + 1)
                case n if n.isdigit() and allow_row_select:
                    return page_items[int(n)]
                case _:
                    pass

    def show_training_summary(self, config: dict[str, Any]) -> bool:
        """
        Presents a final summary of all chosen configurations for user confirmation.

        This is the last step before initiating a long-running process. It gives
        the user a final chance to review their choices (wandb project, overrides,
        etc.) and either confirm or cancel the operation.

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
                TableColumn("left", "bright_cyan",  "Setting", 20),
                TableColumn("left", "bright_white", "Value",   40)
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
