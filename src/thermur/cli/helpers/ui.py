"""
Rich console configuration and utilities for the Thermur CLI.

This module provides a consistent console interface leveraging Rich's
built-in styling and formatting capabilities, encapsulated within the
ThermurUI class.
"""
from omegaconf    import DictConfig
from rich         import progress, box
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
    panels, and progress bars. It uses Hydra configuration objects to define all
    visual styles, ensuring a consistent look and feel across the entire
    application. It is instantiated once and passed to components that need to
    render output.
    """
    def __init__(self, display: DictConfig):
        """
        Initializes the ThermurUI with a configured Rich console.

        Args:
            display : Display configuration containing theme and UI settings.
        """
        self.display = display
        self.console = Console(
            theme          = Theme(display.styles),
            highlight      = False,
            width          = None,
            legacy_windows = False,
        )

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
        fire_colors    = [f"bold {c}" for c in self.display.fire_gradient]
        
        for i, char in enumerate(text):
            color_index = i % len(fire_colors)
            formatted_text.append(char, style=fire_colors[color_index])
        
        return formatted_text

    def create_header_panel(
        self,
        title    : str,
    ) -> Panel:
        """
        Create a styled header panel with optional fire gradient text.
        
        Args:
            title    : Main header text
            
        Returns:
            Formatted header panel
        """
        title_text = Text()
        if "Thermur" in title:
            parts = title.split("Thermur", 1)
            
            if parts[0]:
                title_text.append(parts[0], style="bold bright_white")
            
            title_text.append_text(self.format_fire_gradient_text("Thermur"))
            
            if parts[1]:
                title_text.append(parts[1], style="bold bright_white")
        else:
            title_text.append(title, style="bold bright_white")
        
        return Panel(
            Align.center(title_text),
            border_style = self.display.styles['thermal'],
            padding      = (1, 3),
            expand       = True,
        )

    def print_header(self, title: str):
        """
        Print a styled header panel with fire gradient for "Thermur".
        
        Args:
            title    : Main header text
        """
        self.console.print()
        panel = self.create_header_panel(title)
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
        self.console.print(Rule(title, style=style, align="center"))
        self.console.print()
    
    def print_major_section(
        self,
        title : str,
        style : str = "bright_cyan"
    ):
        """
        Print a major section divider with enhanced styling.
        
        Creates a prominent visual separator for major sections with
        bold formatting and double lines.
        
        Args:
            title : Section title
            style : Style to apply to the rule (default: bright_cyan)
        """
        self.console.print()
        self.console.print(
            Rule(
                title      = f"[bold]{title}[/bold]", 
                style      = style, 
                characters = "═", 
                align      = "center"
            )
        )
        self.console.print()
    
    def print_minor_section(
        self,
        title : str,
        style : str = "grey70"
    ):
        """
        Print a minor section divider with subtle styling.
        
        Creates a subtle visual separator for subsections within major sections.
        
        Args:
            title : Section title  
            style : Style to apply to the rule (default: grey70)
        """
        self.console.print()
        self.console.print(
            Rule(
                title      = title, 
                style      = style, 
                characters = "─", 
                align      = "center"
            )
        )
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
        config = self.display.message_types.get(
            msg_type, 
            self.display.message_types["info"]
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
            f"  [{self.display.styles['muted']}]{description}:[/]"
        )
        self.console.print(
            f"  [bold accent]$ {command}[/]"
        )
        
        if note:
            self.console.print(
                f"  [{self.display.styles['dim']}]  {note}[/]"
            )
        
        self.console.print()

    def print_config_value(
        self,
        key   : str,
        value : str,
        desc  : str | None = None,
        align_width : int = 0
    ):
        """
        Print a configuration key-value pair.
        
        Formats configuration information in a consistent way for
        improved readability.
        
        Args:
            key         : Configuration key
            value       : Configuration value
            desc        : Optional description
            align_width : Width to align the key to (for vertical alignment)
        """
        if align_width:
            key_formatted = f"{key:<{align_width}}"
        else:
            key_formatted = key
            
        if desc:
            self.console.print(
                f"  [{self.display.styles['accent']}]{key_formatted}[/] = "
                f"[white]{value}[/]  "
                f"[{self.display.styles['dim']}]# {desc}[/]"
            )

        else:
            self.console.print(
                f"  [{self.display.styles['accent']}]{key_formatted}[/] = "
                f"[white]{value}[/]"
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
        if url:
            self.console.print(
                f"[{self.display.styles['flock']}]🎨 Dashboard: "
                f"[link={url}]{url}[/link][/]"
            )

        else:
            self.console.print(
                f"[{self.display.styles['flock']}]🎨 Project: "
                f"[{self.display.styles['info']}]{project}[/][/]"
            )


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
                spinner_name = "dots",
                style        = self.display.progress_style,
            ),

            progress.TextColumn(
                "[{0}]{{task.description}}[/{0}]".format(self.display.progress_style)
            ),
            progress.BarColumn(
                bar_width      = 30,
                complete_style = "bright_red",
                style          = self.display.progress_style,
            ),

            progress.TaskProgressColumn(),
            "•",

            progress.TimeElapsedColumn(),
            "•",

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
            border_style = "bright_blue"
        
        # Extract box from kwargs if present to avoid conflict
        box_style = kwargs.pop('box', getattr(box, "MINIMAL", box.MINIMAL))
            
        table = Table(
            title        = title,
            title_style  = "bold bright_cyan",
            header_style = "bold bright_blue",
            border_style = border_style,
            box          = box_style,
            show_edge    = show_edge,
            padding      = (0, 1),
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

    def create_system_table(
        self,
        system_info : dict
    ) -> Table:
        """
        Create a Rich table with system information.

        Generates a comprehensive diagnostic table by combining static component
        definitions with dynamic rendering logic, both sourced from constants.
        The table uses visual indicators and progress bars to make the system
        status immediately apparent.

        Args:
            system_info : A dictionary of system details from 
                          `system.get_system_info()`.

        Returns:
            A formatted Rich table containing system diagnostics.
        """
        settings = dict(self.display.system_table_settings)
        # Remove parameters that are handled by create_aligned_table itself
        for param in ['box', 'title_style', 'border_style']:
            if param in settings:
                del settings[param]
            
        table = self.create_aligned_table(
            columns = [
                (c["header"], c["style"], c["width"], "left") 
                for c in self.display.system_table_columns
            ],
            **settings,
        )

        for key, title in self.display.system_components.items():
            logic = self.display.system_logic[key]
            
            if logic.get("is_resource"):
                # Get thresholds
                thresholds = (4, 8) if key == "memory" else (5, 20)
                progress_bar, details_text = self.format_resource_display(
                    available_gb = system_info.get(logic.get("available"), 0),
                    total_gb     = system_info.get(logic.get("total"), 0),
                    thresholds   = thresholds,
                )
                value = f"{progress_bar}\n{details_text}"

            else:
                # For non-resource items, format the value using the format string
                raw_value  = system_info.get(logic.get("key", key))
                format_str = logic.get("format", "{}")
                
                if key == "cuda":
                    if raw_value:
                        value = Text("✅ Available", style="bold green")
                    else:
                        value = Text("Not Available", style="bold yellow")

                elif key == "gpu":
                    if raw_value and raw_value != "N/A":
                        value = Text(str(raw_value), style="bold green")
                    else:
                        value = Text("Not Detected", style="dim")

                else:
                    if raw_value is not None:
                        value = Text(format_str.format(raw_value), no_wrap=True)
                    else:
                        value = Text("Not Available", style="dim")
                
            table.add_row(Text(title), value)

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
            f"  • {item}" for item in items
        )
        content = f"[{self.display.styles['info']}]{title}:[/]\n{example_text}"
        return Panel(
            content,
            border_style = "bright_blue",
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
            border_style = "bright_blue",
            padding      = (1, 2),
            renderable   = Syntax(
                code         = code,
                lexer        = "yaml",
                theme        = "monokai",
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
        style = self.display.styles['warning']

        return Panel(
            border_style = style, 
            padding      = (1, 2),
            renderable   = (
                f"[bold {style}]{title}[/]\n\n" +
                "\n".join(f"• {i}" for i in issues)
            )
        )

    def create_ready_panel(self, title: str, subtitle: str) -> Panel:
        """
        Assembles a success-themed panel to confirm a 'ready' state to the user.
        
        Provides positive visual feedback, for instance, before starting a
        long-running process like model training.
        
        Args:
            title    : The main title for the panel (e.g., "Ready to train!").
            subtitle : The subtitle text to display below the title.
            
        Returns:
            A configured Panel object with success styling.
        """
        style   = self.display.styles['success']
        content = Align.left(
            f"[bold {style}]{title}[/]\n"
            f"[{self.display.styles['muted']}]{subtitle}[/]",
            vertical="middle",
        )
        return Panel(
            content, 
            border_style = style, 
            padding      = (1, 3)
        )


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
            length = self.display.progress_bar_length
            
        return (
            f"[{color}]"
            f"{'█' * int(used_fraction * length)}"
            f"[/{color}][{self.display.progress_unfilled_color}]"
            f"{'░' * (length - int(used_fraction * length))}"
            f"[/{self.display.progress_unfilled_color}]"
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
            "red"     if available_gb < low_thresh else
            "yellow"  if available_gb < med_thresh else
            "bright_green"
        )
        
        progress_bar = self.create_progress_bar(color, used_fraction)
        details_text = self.display.resource_details_template.format(
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
