"""
Rich console configuration and utilities for the Thermur CLI.

This module provides a consistent console interface leveraging Rich's
built-in styling and formatting capabilities, encapsulated within the
ThermurUI class.
"""
from .constants    import CLIConstants
from rich          import progress
from rich.align    import Align
from rich.box      import MINIMAL, ROUNDED
from rich.console  import Console
from rich.panel    import Panel
from rich.rule     import Rule
from rich.table    import Table
from rich.text     import Text
from rich.theme    import Theme


class ThermurUI:
    """
    Manages all Rich console rendering for the Thermur CLI.
    """
    def __init__(self):
        """
        Initializes the ThermurUI with a configured Rich console.
        """
        self.console = Console(
            theme     = Theme(CLIConstants.Theme.STYLES),
            highlight = False
        )

    @staticmethod
    def format_fire_gradient_text(text: str) -> Text:
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
        fire_colors    = [f"bold {c}" for c in CLIConstants.Theme.FIRE_GRADIENT]
        formatted_text = Text()
        
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
                title_text.append(parts[0], style=CLIConstants.UI.HEADER_TEXT_STYLE)
            
            title_text.append_text(self.format_fire_gradient_text("Thermur"))
            
            if parts[1]:
                title_text.append(parts[1], style=CLIConstants.UI.HEADER_TEXT_STYLE)
        else:
            title_text.append(title, style=CLIConstants.UI.TITLE_TEXT_STYLE)
        
        if subtitle:
            title_text.append("\n")
            title_text.append(subtitle, style=CLIConstants.UI.SUBTITLE_TEXT_STYLE)
        
        return Panel(
            Align.center(title_text),
            border_style = CLIConstants.UI.PANEL_BORDER_STYLE,
            box          = ROUNDED,
            padding      = CLIConstants.UI.PANEL_PADDING,
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

    def print_section(self, title: str, style: str = "accent"):
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
        config = CLIConstants.Messages.TYPES.get(
            msg_type, 
            CLIConstants.Messages.TYPES["info"]
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
        self.console.print(f"  [muted]{description}:[/muted]")
        self.console.print(f"  [bold accent]$ {command}[/bold accent]")
        
        if note:
            self.console.print(f"  [dim italic]  {note}[/dim italic]")
        
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
                f"  [accent]{key}[/accent] = [bright_white]{value}[/bright_white]"
                f"  [dim]# {desc}[/dim]"
            )

        else:
            self.console.print(
                f"  [accent]{key}[/accent] = [bright_white]{value}[/bright_white]"
            )

    def print_status_badge(
        self,
        label  : str,
        status : str,
        style  : str = "success"
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
        if url and CLIConstants.UI.WANDB_URL_PLACEHOLDER not in url:
            self.console.print(f"[swarm]📊 Dashboard: [link={url}]{url}[/link][/swarm]")

        else:
            self.console.print(f"[swarm]📊 Project: [bright_cyan]{project}[/bright_cyan][/swarm]")

    def print_training_tips(self):
        """
        Print helpful training tips.
        
        Shows useful information to help users get the most out of their
        training runs, using tips from centralized constants.
        """
        self.print_section(
            CLIConstants.Tips.SECTION_TITLE, 
            style=CLIConstants.Tips.SECTION_STYLE
        )
        
        for tip in CLIConstants.Tips.TRAINING:
            bullet = f"[{CLIConstants.Tips.BULLET_STYLE}]{CLIConstants.UI.BULLET_CHAR}"
            self.console.print(f"  {bullet}[/{CLIConstants.Tips.BULLET_STYLE}] {tip['desc']}")
            self.console.print(f"    [dim]{tip['command']}[/dim]")
        
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
                spinner_name = CLIConstants.UI.PROGRESS_SPINNER,
                style        = "thermal"
            ),

            progress.TextColumn("[thermal]{task.description}[/thermal]"),
            progress.BarColumn(
                bar_width      = CLIConstants.UI.PROGRESS_BAR_WIDTH,
                complete_style = "bright_red",
                style          = "thermal"
            ),

            progress.TaskProgressColumn(),
            CLIConstants.UI.BULLET_CHAR,

            progress.TimeElapsedColumn(),
            CLIConstants.UI.BULLET_CHAR,

            progress.MofNCompleteColumn(),
            console   = self.console,
            expand    = False,
            transient = False,
        )

    def create_aligned_table(
        self,
        title        : str,
        columns      : list[tuple[str, str, int, str]],
        border_style : str = None,
        show_edge    : bool = False,
        expand       : bool = False,
    ) -> Table:
        """
        Create a properly aligned table with consistent thermal styling.
        
        Constructs a Rich Table with standardized formatting and alignment,
        ensuring visual consistency across all CLI components.
        
        Args:
            title        : Table title displayed at the top
            columns      : List of (title, style, width, align) tuples
            border_style : Rich style string for table borders
            show_edge    : Whether to display table edge borders
            expand       : Whether to expand table to full terminal width
            
        Returns:
            Configured Table instance ready for content population
        """
        if border_style is None:
            border_style = CLIConstants.UI.TABLE_BORDER_STYLE
            
        table = Table(
            title        = title,
            title_style  = CLIConstants.UI.TABLE_TITLE_STYLE,
            header_style = CLIConstants.UI.TABLE_HEADER_STYLE,
            border_style = border_style,
            box          = MINIMAL,
            show_edge    = show_edge,
            padding      = CLIConstants.UI.TABLE_PADDING,
            expand       = expand,
        )
        
        for col_title, style, width, align in columns:
            table.add_column(
                col_title,
                style   = style,
                width   = width,
                justify = align,
            )
        
        return table

    def create_feature_table(self):
        """
        Create a table showcasing Thermur features.
        
        Uses the centralized table creation utility with consistent
        styling and the feature list from constants.
        
        Returns:
            A formatted table with feature information
        """
        table = self.create_aligned_table(
            title   = CLIConstants.Features.TABLE_TITLE,
            columns = CLIConstants.Features.TABLE_COLUMNS,
        )
        
        for feature in CLIConstants.Features.LIST:
            table.add_row(feature["name"], feature["desc"], feature["status"])
        
        return table

    @staticmethod
    def create_progress_bar(
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
            length = CLIConstants.UI.PROGRESS_BAR_DEFAULT_LENGTH
            
        return (
            f"[{color}]"
            f"{CLIConstants.UI.FILLED_CHAR * int(used_fraction * length)}"
            f"[/{color}][{CLIConstants.UI.PROGRESS_UNFILLED_COLOR}]"
            f"{CLIConstants.UI.UNFILLED_CHAR * (length - int(used_fraction * length))}"
            f"[/{CLIConstants.UI.PROGRESS_UNFILLED_COLOR}]"
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
            CLIConstants.UI.RESOURCE_COLOR_CRITICAL if available_gb < low_thresh else
            CLIConstants.UI.RESOURCE_COLOR_WARNING  if available_gb < med_thresh else
            CLIConstants.UI.RESOURCE_COLOR_GOOD
        )
        
        progress_bar = self.create_progress_bar(color, used_fraction)
        details_text = (
            f"[white]{available_gb:.1f}{unit} "
            f"free of {total_gb:.1f}{unit}[/white]"
        )
        
        return progress_bar, details_text