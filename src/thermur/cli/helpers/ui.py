"""
Rich console configuration and utilities for the Thermur CLI.

This module provides a consistent console interface leveraging Rich's
built-in styling and formatting capabilities, encapsulated within the
ThermurUI class.
"""
from rich         import progress
from rich.align   import Align
from rich.console import Console
from rich.panel   import Panel
from rich.rule    import Rule
from rich.syntax  import Syntax
from rich.table   import Table
from rich.text    import Text
from rich.theme   import Theme


class ThermurUI:
    """
    Manages all console rendering for the Thermur CLI using the Rich library.

    This class centralizes the creation of styled components like headers, tables,
    panels, and progress bars. It receives a `CLIConstants` object during
    initialization, which it uses to configure all visual styles. This ensures a
    consistent look and feel across the entire application. It is instantiated
    once in `app.py` and passed to other components that need to render output.
    """
    def __init__(self, constants):
        """
        Initializes the ThermurUI with a configured Rich console.

        Args:
            constants: An instance of `CLIConstants` containing all static
                       style and text data for the CLI.
        """
        self.constants = constants
        self.console   = Console(
            theme     = Theme(self.constants.Theme.STYLES),
            highlight = False
        )

    @staticmethod
    def format_fire_gradient_text(self, text: str) -> Text:
        """
        Format text with the distinctive Thermur fire gradient colors.
        
        Applies the signature thermal gradient to create the distinctive visual
        effect used for "Thermur" branding. The gradient transitions from deep
        red through orange to bright yellow.
        
        Args:
            text : Input text to apply the fire gradient styling
            
        Returns:
            Rich Text object with fire gradient colors applied per character
        """
        formatted_text = Text()
        fire_colors    = self.constants.Theme.FIRE_GRADIENT_STYLED
        
        for i, char in enumerate(text):
            color_index = i % len(fire_colors)
            formatted_text.append(char, style=fire_colors[color_index])
        
        return formatted_text

    def create_header_panel(
        self,
        title    : str,
        subtitle : str | None = None,
    ) -> Panel:
        """
        Create a styled header panel with optional fire gradient text.
        
        Args:
            title    : Main header text
            subtitle : Optional subtitle text
            
        Returns:
            Formatted header panel
        """
        title_text = Text()
        
        # Apply fire gradient to "Thermur" if present
        if "Thermur" in title:
            parts = title.split("Thermur", 1)
            
            if parts[0]:
                title_text.append(parts[0], style=self.constants.UI.HEADER_TEXT_STYLE)
            
            title_text.append_text(self.format_fire_gradient_text("Thermur"))
            
            if parts[1]:
                title_text.append(parts[1], style=self.constants.UI.HEADER_TEXT_STYLE)
        else:
            title_text.append(title, style=self.constants.UI.TITLE_TEXT_STYLE)
        
        if subtitle:
            title_text.append("\n")
            title_text.append(subtitle, style=self.constants.UI.SUBTITLE_TEXT_STYLE)
        
        return Panel(
            Align.center(title_text),
            border_style = self.constants.UI.PANEL_BORDER_STYLE,
            box          = self.constants.UI.PANEL_BOX,
            padding      = self.constants.UI.PANEL_PADDING,
        )

    def print_header(self, title: str, subtitle: str | None = None):
        """
        Print a styled header panel with fire gradient for "Thermur".
        
        Args:
            title    : Main header text
            subtitle : Optional subtitle text
        """
        self.console.print()
        panel = self.create_header_panel(title, subtitle)
        self.console.print(panel)
        self.console.print()

    def print_section(
        self,
        title : str,
        style : str
    ):
        """
        Print a section divider with title.
        
        Creates a visual separator between different sections of output
        using Rich's Rule component.
        
        Args:
            title : Section title
            style : Style to apply to the rule
        """
        self.console.print()
        self.console.print(Rule(f" {title} ", style=style))
        self.console.print()

    def print_message(self, message: str, msg_type: str = "info"):
        """
        Print a styled message with appropriate icon and formatting.
        
        This function provides a unified interface for all message types,
        using the centralized message configurations.
        
        Args:
            message  : The message text to display
            msg_type : Type of message (info, warning, error, etc.)
        """
        config = self.constants.Messages.TYPES.get(
            msg_type, 
            self.constants.Messages.TYPES["info"]
        )
        
        self.console.print(
            f"[{config['style']}]{config['icon']} {message}[/{config['style']}]"
        )

    def print_command_example(
        self,
        description : str,
        command     : str,
        note        : str | None = None
    ):
        """
        Print a formatted command example.
        
        Shows command examples in a visually distinct way to help users
        understand how to use the CLI effectively.
        
        Args:
            description : What the command does
            command     : The actual command to run
            note        : Optional note about the command
        """
        self.console.print(
            f"  [{self.constants.UI.MUTED_STYLE}]{description}:[/]"
        )
        self.console.print(
            f"  [{self.constants.UI.COMMAND_STYLE}]$ {command}[/]"
        )
        
        if note:
            self.console.print(
                f"  [{self.constants.UI.DIM_STYLE}]  {note}[/]"
            )
        
        self.console.print()

    def print_config_value(
        self,
        key   : str,
        value : str,
        desc  : str | None = None
    ):
        """
        Print a configuration key-value pair.
        
        Formats configuration information in a consistent way for
        improved readability.
        
        Args:
            key   : Configuration key
            value : Configuration value
            desc  : Optional description
        """
        if desc:
            self.console.print(
                f"  [{self.constants.Theme.STYLES['accent']}]{key}[/] = "
                f"[{self.constants.UI.WHITE_STYLE}]{value}[/]  "
                f"[{self.constants.Theme.STYLES['dim']}]# {desc}[/]"
            )

        else:
            self.console.print(
                f"  [{self.constants.Theme.STYLES['accent']}]{key}[/] = "
                f"[{self.constants.UI.WHITE_STYLE}]{value}[/]"
            )

    def print_status_badge(
        self,
        label  : str,
        status : str,
        style  : str
    ):
        """
        Print a status badge.
        
        Creates a compact status indicator for various system states
        with consistent formatting.
        
        Args:
            label  : Badge label
            status : Status text
            style  : Style to apply
        """
        badge = f"[{style}][ {label}: {status} ][/{style}]"
        self.console.print(badge)

    def print_wandb_info(self, project: str, url: str | None = None):
        """
        Print wandb project information.
        
        Displays wandb integration status and provides links to monitoring
        dashboards when available.
        
        Args:
            project : The wandb project name
            url     : Optional URL to the project dashboard
        """
        icon = self.constants.UI.WANDB_ICON
        if url and self.constants.UI.WANDB_URL_PLACEHOLDER not in url:
            self.console.print(
                f"[{self.constants.Theme.STYLES['swarm']}]{icon} Dashboard: "
                f"[link={url}]{url}[/link][/]"
            )

        else:
            self.console.print(
                f"[{self.constants.Theme.STYLES['swarm']}]{icon} Project: "
                f"[{self.constants.UI.CYAN_STYLE}]{project}[/][/]"
            )

    def print_training_tips(self):
        """
        Print helpful training tips.
        
        Shows useful information to help users get the most out of their
        training runs, using tips from centralized constants.
        """
        self.print_section(
            title = self.constants.Tips.SECTION_TITLE, 
            style = self.constants.Tips.SECTION_STYLE
        )
        
        for tip in self.constants.Tips.TRAINING:
            self.console.print(
                f"  [{self.constants.Tips.BULLET_STYLE}]"
                f"{self.constants.UI.BULLET_CHAR}"
                f"[/{self.constants.Tips.BULLET_STYLE}] {tip['desc']}"
            )
            self.console.print(
                f"    [{self.constants.Theme.STYLES['dim']}]{tip['command']}[/]"
            )
        
        self.console.print()

    def create_thermal_progress(self) -> progress.Progress:
        """
        Create a thermal-themed progress bar for long-running operations.
        
        Constructs a Rich Progress instance with thermal styling and comprehensive
        progress tracking including elapsed time, completion percentage, and
        task descriptions.
        
        Returns:
            Configured Progress instance with thermal styling and timing components
        """
        return progress.Progress(

            progress.SpinnerColumn(
                spinner_name = self.constants.UI.PROGRESS_SPINNER,
                style        = self.constants.UI.PROGRESS_STYLE,
            ),

            progress.TextColumn(
                f"[{self.constants.UI.PROGRESS_STYLE}]"
                "{task.description}"
                f"[/{self.constants.UI.PROGRESS_STYLE}]"
            ),
            progress.BarColumn(
                bar_width      = self.constants.UI.PROGRESS_BAR_WIDTH,
                complete_style = self.constants.UI.PROGRESS_COMPLETE_STYLE,
                style          = self.constants.UI.PROGRESS_STYLE,
            ),

            progress.TaskProgressColumn(),
            self.constants.UI.BULLET_CHAR,

            progress.TimeElapsedColumn(),
            self.constants.UI.BULLET_CHAR,

            progress.MofNCompleteColumn(),
            console   = self.console,
            expand    = False,
            transient = False,
        )

    def create_aligned_table(
        self,
        columns      : list[tuple[str, str, int, str]],
        title        : str = "",
        border_style : str = None,
        show_edge    : bool = False,
        expand       : bool = False,
        **kwargs
    ) -> Table:
        """
        Create a properly aligned table with consistent thermal styling.
        
        Constructs a Rich Table with standardized formatting and alignment,
        ensuring visual consistency across all CLI components.
        
        Args:
            columns      : List of (title, style, width, align) tuples
            title        : Table title displayed at the top
            border_style : Rich style string for table borders
            show_edge    : Whether to display table edge borders
            expand       : Whether to expand table to full terminal width
            **kwargs     : Additional keyword arguments for Rich Table
            
        Returns:
            Configured Table instance ready for content population
        """
        if border_style is None:
            border_style = self.constants.UI.TABLE_BORDER_STYLE
            
        table = Table(
            title        = title,
            title_style  = self.constants.UI.TABLE_TITLE_STYLE,
            header_style = self.constants.UI.TABLE_HEADER_STYLE,
            border_style = border_style,
            box          = self.constants.UI.TABLE_BOX,
            show_edge    = show_edge,
            padding      = self.constants.UI.TABLE_PADDING,
            expand       = expand,
            **kwargs,
        )
        
        for col_title, style, width, align in columns:
            table.add_column(
                col_title,
                style   = style,
                width   = width,
                justify = align,
            )
        
        return table

    def create_system_table(self, system_info: dict) -> Table:
        """
        Create a Rich table with system information.

        Generates a comprehensive diagnostic table by combining static component
        definitions with dynamic rendering logic, both sourced from constants.
        The table uses visual indicators and progress bars to make the system
        status immediately apparent.

        Args:
            system_info: A dictionary of system details from `system.get_system_info()`.

        Returns:
            A formatted Rich table containing system diagnostics.
        """
        table = self.create_aligned_table(
            title   = self.constants.System.TABLE_TITLE,
            columns = [(c["header"], c["style"], c["width"], "left") for c in self.constants.System.TABLE_COLUMNS],
            **self.constants.System.TABLE_SETTINGS,
        )

        for key, title in self.constants.System.SYSTEM_COMPONENTS.items():
            logic = self.constants.System.SYSTEM_LOGIC[key]
            
            if logic.get("is_resource"):
                progress_bar, details_text = self.format_resource_display(
                    available_gb = system_info.get(f"{key}_available", 0),
                    total_gb     = system_info.get(f"{key}_total", 0),
                    thresholds   = getattr(self.constants.SystemChecks, f"{str.upper(key)}_THRESHOLDS"),
                )
                details = f"{progress_bar}\n{details_text}"

            else:
                details = logic["details"](system_info)
                
            table.add_row(
                title, 
                logic["status"](system_info), 
                details
            )

        return table

    def create_examples_panel(
        self,
        items : list[str],
        title : str = "Examples"
    ) -> Panel:
        """
        Constructs a Rich Panel to consistently display a list of examples.
        
        This method is used to show users potential inputs or options, such
        as example project names, in a visually distinct and standardized way.
        
        Args:
            items : A list of strings to display as bullet points.
            title : The title for the examples list.
            
        Returns:
            A configured Panel object.
        """
        example_text = "\n".join(
            f"  {self.constants.UI.BULLET_CHAR} {item}" for item in items
        )
        content = f"[{self.constants.UI.CYAN_STYLE}]{title}:[/]\n{example_text}"
        return Panel(
            content,
            border_style = self.constants.UI.PANEL_BORDER_STYLE,
            padding      = (0, 2),
        )

    def create_syntax_panel(self, code: str, title: str) -> Panel:
        """
        Builds a Rich Panel containing a Syntax object for displaying code.
        
        This is ideal for showing users configuration examples or command
        syntax in a properly highlighted and readable format.
        
        Args:
            code  : The string of code to format.
            title : The title of the panel.
            
        Returns:
            A configured Panel object containing a Syntax object.
        """
        return Panel(
            title        = title,
            border_style = self.constants.UI.PANEL_BORDER_STYLE,
            padding      = (1, 2),
            renderable   = Syntax(
                code         = code,
                lexer        = "yaml",
                theme        = self.constants.UI.SYNTAX_THEME,
                line_numbers = False,
            ),
        )

    def create_warning_panel(self, title: str, issues: list[str]) -> Panel:
        """
        Generates a styled warning panel to present a list of issues or problems.
        
        Uses a distinct warning color scheme to draw user attention to
        important system checks, validation errors, or other non-blocking issues.
        
        Args:
            title  : The main title for the warning (e.g., "Issues Detected").
            issues : A list of strings detailing the issues to be listed.
            
        Returns:
            A configured Panel object with warning styling.
        """
        style = self.constants.Theme.STYLES['warning']

        return Panel(
            border_style = style, 
            padding      = (1, 2),
            renderable   = (
                f"[bold {style}]{title}[/]\n\n" +
                "\n".join(f"{self.constants.UI.BULLET_CHAR} {i}" for i in issues)
            )
        )

    def create_ready_panel(self, title: str, subtitle: str) -> Panel:
        """
        Assembles a success-themed panel to confirm a 'ready' state to the user.
        
        Provides positive visual feedback, for instance, before starting a
        long-running process like model training.
        
        Args:
            title    : The main title for the panel (e.g., "Ready to Train!").
            subtitle : The subtitle text to display below the title.
            
        Returns:
            A configured Panel object with success styling.
        """
        style = self.constants.Theme.STYLES['success']
        content = Align.center(
            f"[bold {style}]{title}[/]\n"
            f"[{self.constants.UI.MUTED_STYLE}]{subtitle}[/]",
            vertical="middle",
        )
        return Panel(
            content, 
            border_style = style, 
            padding      = self.constants.UI.PANEL_PADDING
        )

    def create_feature_table(self):
        """
        Create a table showcasing Thermur features.
        
        Uses the centralized table creation utility with consistent
        styling and the feature list from constants.
        
        Returns:
            A formatted table with feature information
        """
        table = self.create_aligned_table(
            title   = self.constants.Features.TABLE_TITLE,
            columns = self.constants.Features.TABLE_COLUMNS,
        )
        
        for feature in self.constants.Features.LIST:
            table.add_row(feature["name"], feature["desc"], feature["status"])
        
        return table

    @staticmethod
    def create_progress_bar(
        self,
        color         : str,
        used_fraction : float,
        length        : int = None,
    ) -> str:
        """
        Create a string-based progress bar using Rich markup for resource display.
        
        Constructs a visual progress bar using Unicode block characters and Rich
        styling. Color coding helps users quickly assess system health.
        
        Args:
            color         : Rich color name for the filled portion of the bar
            used_fraction : Utilization ratio from 0.0 (empty) to 1.0 (full)
            length        : Total character width of the progress bar
            
        Returns:
            Rich markup string representing the styled progress bar
        """
        if length is None:
            length = self.constants.System.PROGRESS_BAR_LENGTH
            
        return (
            f"[{color}]"
            f"{self.constants.UI.FILLED_CHAR * int(used_fraction * length)}"
            f"[/{color}][{self.constants.UI.PROGRESS_UNFILLED_COLOR}]"
            f"{self.constants.UI.UNFILLED_CHAR * (length - int(used_fraction * length))}"
            f"[/{self.constants.UI.PROGRESS_UNFILLED_COLOR}]"
        )

    def format_resource_display(
        self,
        available_gb : float,
        total_gb     : float,
        thresholds   : tuple[int, int],
        unit         : str = "GB",
    ) -> tuple[str, str]:
        """
        Format system resource display with intelligent color-coded progress bars.
        
        Creates resource utilization display with visual progress indicators and
        precise numerical information. The threshold-based coloring provides
        immediate feedback about resource availability.
        
        Args:
            available_gb : Amount of available resource in gigabytes
            total_gb     : Total system resource capacity in gigabytes
            thresholds   : Tuple of (low_threshold, medium_threshold) in GB
            unit         : Display unit string (typically "GB")
            
        Returns:
            Tuple containing (progress_bar_markup, details_text) for display
        """
        used_fraction = (total_gb - available_gb) / total_gb if total_gb > 0 else 0
        low_thresh, med_thresh = thresholds
        
        color = (
            self.constants.UI.RESOURCE_COLOR_CRITICAL if available_gb < low_thresh else
            self.constants.UI.RESOURCE_COLOR_WARNING  if available_gb < med_thresh else
            self.constants.UI.RESOURCE_COLOR_GOOD
        )
        
        progress_bar = self.create_progress_bar(color, used_fraction)
        details_text = self.constants.UI.RESOURCE_DETAILS_TEMPLATE.format(
            available_gb, unit, total_gb, unit
        )
        
        return progress_bar, details_text
    
    @staticmethod
    def get_field_type_string(field_type: any) -> str:
        """
        Get a readable type string for a Pydantic model field.

        Args:
            field_type: The type annotation to inspect.

        Returns:
            A human-readable type string (e.g., "int", "list[str]").
        """
        from typing import get_args, get_origin
        
        if origin := get_origin(field_type):
            args = get_args(field_type)
            if not args:
                return origin.__name__
            
            inner_types = ", ".join(ThermurUI.get_field_type_string(arg) for arg in args)
            return f"{origin.__name__}[{inner_types}]"
        
        return getattr(field_type, "__name__", str(field_type))

    @staticmethod
    def get_component_description(component: any) -> str:
        """
        Generate a concise, one-line description for a configuration component.
        
        Args:
            component: The configuration component to describe.
            
        Returns:
            A human-readable description string.
        """
        from hydra_zen import get_target

        if doc := getattr(component, "__doc__", None):
            return doc.strip().split("\n")[0]
        
        if target := getattr(component, "_target_", None):
            if target_obj := get_target(component):
                return f"{target_obj.__module__}.{target_obj.__name__}"
        
        return f"Configuration object ({type(component).__name__})"
