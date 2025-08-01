"""
Rich console configuration and utilities for the Thermur CLI.

This module provides a consistent console interface leveraging Rich's
built-in styling and formatting capabilities, encapsulated within the
ThermurUI class.
"""
from .system      import SystemInspector
from collections  import Counter
from config.cli   import DisplayModel
from config.types import FileInfo, TableColumn
from pathlib      import Path
from rich         import box, progress
from rich.align   import Align
from rich.console import Console
from rich.panel   import Panel
from rich.rule    import Rule
from rich.syntax  import Syntax
from rich.table   import Table
from rich.text    import Text
from rich.theme   import Theme
from typing       import Any, Literal, Sequence


class ThermurUI:
    """
    Manages all console rendering for the Thermur CLI using the Rich library.

    This class centralizes the creation of styled components like headers, tables,
    panels, and progress bars. It uses Hydra configuration objects to define all
    visual styles, ensuring a consistent look and feel across the entire
    application. It is instantiated once and passed to components that need to
    render output.
    """
    
    def __init__(self, display: DisplayModel):
        """
        Initializes the ThermurUI with a configured Rich console.

        Args:
            display  : Display configuration containing theme and UI settings.
        """
        self.display       = display
        self.console       = Console(
            theme          = Theme(display.styles),
            highlight      = False,
            width          = None,
            legacy_windows = False,
        )

    def _create_header_panel(self, title: str) -> Panel:
        """
        Create a styled header panel with fire gradient for "Thermur".
        
        Creates a visually striking header panel with special formatting
        for the word "Thermur" using the signature fire gradient effect.
        
        Args:
            title: Main header text
            
        Returns:
            Formatted header panel
        """
        title_text = Text()
        if "Thermur" in title:
            parts = title.split("Thermur", 1)
            
            if parts[0]:
                title_text.append(parts[0], "bold bright_white")
            
            title_text.append_text(self._format_fire_gradient_text("Thermur"))
            
            if parts[1]:
                title_text.append(parts[1], "bold bright_white")
        else:
            title_text.append(title, "bold bright_white")
        
        return Panel(
            Align.center(title_text),
            border_style = self.display.styles['thermal'],
            padding      = (1, 3),
            expand       = True,
        )

    def _create_progress_bar(
        self,
        color         : str,
        used_fraction : float,
        length        : int | None = None
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
            f"[/{color}][{self.display.styles['dim']}]"
            f"{'░' * (length - int(used_fraction * length))}"
            f"[/{self.display.styles['dim']}]"
        )
    
    def _create_status_indicator(self, text: str, is_available: bool) -> Text:
        """
        Create a status indicator Text with availability-based styling.
        
        Used for system component display where available/present items
        are shown in green and unavailable/missing items are dimmed.
        
        Args:
            text         : The text to display
            is_available : Whether the component/resource is available
            
        Returns:
            Text object styled green for available, dim for unavailable
        """
        return Text(text, "bold green" if is_available else "dim")

    def _format_fire_gradient_text(self, text: str) -> Text:
        """
        Format text with the distinctive Thermur fire gradient colors.
        
        Applies the signature thermal gradient to create the distinctive visual
        effect used for "Thermur" branding. The gradient transitions from deep
        red through bright orange to yellow, mimicking thermal imaging.
        
        Args:
            text : The text to apply the gradient to
            
        Returns:
            Rich Text object with character-by-character gradient styling
        """
        gradient_text = Text()
        colors = self.display.fire_gradient
        
        for i, char in enumerate(text):
            color = colors[i % len(colors)]
            gradient_text.append(char, f"bold {color}")
            
        return gradient_text

    def _format_resource_display(
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
        
        progress_bar = self._create_progress_bar(color, used_fraction)
        details_text = f"{available_gb:.1f} {unit} / {total_gb:.1f} {unit}"
        
        return progress_bar, details_text
    
    def _format_resource(
        self, 
        available : float, 
        threshold : tuple[int, int],
        total     : float, 
    ) -> str:
        """
        Format a resource with progress bar and details.
        
        Args:
            available : Available amount in GB
            threshold : (warning, critical) thresholds in GB
            total     : Total amount in GB  
            
        Returns:
            Formatted string with progress bar and details
        """
        progress_bar, details_text = self._format_resource_display(
            available_gb = available,
            total_gb     = total,
            thresholds   = threshold,
        )
        return f"{progress_bar} {details_text}"

    def create_aligned_table(
        self,
        columns      : list[TableColumn],
        border_style : str | None = None,
        expand       : bool       = False,
        show_edge    : bool       = False,
        title        : str        = "",
        **kwargs     : Any
    ) -> Table:
        """
        Create a properly aligned table with consistent thermal styling.
        
        Constructs a Rich Table with standardized formatting and alignment,
        ensuring visual consistency across all CLI components.
        
        Args:
            columns      : List of (title, style, width, align) tuples
            border_style : Rich style string for table borders
            expand       : Whether to expand table to full terminal width
            show_edge    : Whether to display table edge borders
            title        : Table title displayed at the top
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
        
        for col in columns:
            table.add_column(
                col.title,
                style   = col.style,
                width   = col.width,
                justify = col.justify,
            )
        
        return table

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
            renderable = f"[bold {style}]{title}[/]\n"
                         f"[{self.display.styles['muted']}]{subtitle}[/]",
            vertical   = "middle"
        )
        return Panel(
            content, 
            border_style = style, 
            padding      = (1, 3)
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

    def create_system_table(self, system_info: dict[str, Any]) -> Table:
        """
        Create a Rich table with system information.
        
        Args:
            system_info: Dictionary of system details from system.get_system_info().
            
        Returns:
            A formatted Rich table containing system diagnostics.
        """
        table = self.create_aligned_table(
            columns = [
                TableColumn(
                    justify = "left",
                    style   = str(c["style"]),
                    title   = str(c["header"]),
                    width   = int(c["width"])
                )
                for c in self.display.system_table_columns
            ],
            show_edge = True
        )
        
        resource_thresholds = {"memory": (4, 8), "disk": (5, 20)}
        version_components  = {"python", "torch", "mujoco", "thermur"}
        
        for key, title in self.display.system_components.items():
            if key in resource_thresholds:
                value = self._format_resource(
                    available = system_info.get(f"{key}_available", 0),
                    total     = system_info.get(f"{key}_total", 0),
                    threshold = resource_thresholds[key]
                )
            elif key == "cuda":
                version = system_info.get("cuda")
                value = self._create_status_indicator(
                    f"✅ Available (v{version})" if version else "Not Available",
                    bool(version)
                )
            elif key == "gpu":
                gpu_name = system_info.get("gpu")
                available = bool(gpu_name and gpu_name != "N/A")
                value = self._create_status_indicator(
                    str(gpu_name) if available else "Not Detected",
                    available
                )
            elif key == "dataset":
                size = system_info.get("dataset_size", 0)
                count = system_info.get("dataset_count", 0)
                text = (
                    f"{size:.1f} GB ({count} files)" if size > 0
                    else "No files downloaded"
                )
                value = Text(text, "white" if size > 0 else "dim")
            elif val := system_info.get(key):
                prefix = "v" if key in version_components else ""
                value = Text(f"{prefix}{val}", no_wrap=True)
            else:
                value = self._create_status_indicator("Not Available", False)
                
            table.add_row(Text(title), value)
            
        return table

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
                style        = self.display.styles['thermal'],
            ),

            progress.TextColumn(
                text_format = (
                    f"[{self.display.styles['thermal']}]"
                    f"{{task.description}}"
                    f"[/{self.display.styles['thermal']}]"
                )
            ),
            progress.BarColumn(
                bar_width        = self.display.progress_bar_length,
                complete_style   = self.display.styles['thermal'],
                finished_style   = self.display.styles['thermal'],
                pulse_style      = self.display.styles['thermal'],
                style            = self.display.styles['dim'],
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
    
    def display_download_summary(
        self,
        available_files : Sequence[FileInfo],
        file_status     : dict[str, str]
    ):
        """
        Display summary statistics for dataset download status.
        
        Shows total dataset size, downloaded size, and file counts
        broken down by status.
        
        Args:
            available_files : List of all available files with size info
            file_status     : Dict mapping filename to status
        """
        statuses       = Counter(file_status.values())
        size_by_status = {
            status: sum(
                f['size'] for f in available_files 
                if file_status.get(f['name']) == status
            )
            for status in ['downloaded', 'incomplete', 'missing']
        }
        
        total_size = sum(f['size'] for f in available_files)
        
        total_gb = total_size / 1e9
        self.print_message(
            message  = f"Total: {len(available_files)} files ({total_gb:.1f} GB)",
            msg_type = "info"
        )
        
        status_info = [
            ('downloaded', "Downloaded: {} files ({:.1f} GB)",     "success"),
            ('incomplete', "Incomplete: {} files ({:.1f} GB)",     "warning"),
            ('missing',    "Not downloaded: {} files ({:.1f} GB)", "info")
        ]
        
        for status_type, template, msg_type in status_info:
            if count := statuses[status_type]:
                size_gb = size_by_status[status_type] / 1e9
                self.print_message(f"{template.format(count, size_gb)}", msg_type)

    def display_download_table(
        self,
        available_files : list[FileInfo],
        file_status     : dict[str, str],
        title           : str  = "Available Files"
    ):
        """
        Display a paginated table of files with their download status.
        
        Shows up to 10 files with status indicators and selection numbers (0-9).
        
        Args:
            available_files : List of file info dictionaries with 'name' and 'size'
            file_status     : Dict mapping filename to download status
                              ('downloaded', 'incomplete', or 'missing')
            title           : Table title
        """
        status_symbols = {
            'downloaded' : Text("✅", "green"),
            'incomplete' : Text("⚠️", "yellow"), 
            'missing'    : Text(" ",  "dim")
        }
        
        columns = [
            TableColumn("right",  "bright_cyan", "#",      4),
            TableColumn("center", "green",       "Status", 8),
            TableColumn("left",   "cyan",        "File",   50),
            TableColumn("right",  "yellow",      "Size",   10)
        ]
        
        table = self.create_aligned_table(columns, title=title)
        
        for i, file_info in enumerate(available_files):
            name   = file_info['name']
            size   = file_info['size']
            status = file_status.get(name, 'missing')
            row: list[Any] = [str(i), status_symbols[status], name, f"{size / 1e9:.1f} GB"]
                
            table.add_row(*row)
            
        self.console.print()
        self.console.print(table)

    def display_panel(
        self, 
        content      : str | Table | Panel, 
        border_style : str = "dim",
        title        : str = "" 
    ):
        """
        Creates and displays a styled panel with the given content.
        
        For string content, wraps it in a Panel. For Rich objects (Table, Panel),
        displays them directly.
        
        Args:
            content      : The content to display (string, Table, or Panel).
            border_style : Style for the panel border (default: "dim").
            title        : Optional title for the panel.
        """
        self.console.print(
            Panel(
                border_style = border_style,
                padding      = (1, 2),
                renderable   = content,
                title        = title
            )
        )
    
    def display_status_legend(self):
        """
        Display the legend for run status indicators.
        
        Shows a formatted legend explaining the meaning of status symbols
        used in run listings: ✓ for complete, ◎ for dry run, ... for incomplete.
        """
        self.console.print(
            "ℹ️  Status: "
            "[bold green](✓) Complete[/], "
            "[bold cyan](◎) Dry Run[/], "
            "[bold yellow](...) Incomplete[/]"
        )

    def display_system_validation(self, system: SystemInspector):
        """
        Display comprehensive system validation information.
        
        Shows system information and wandb status in a formatted display.
        Used by both train and validate commands for consistent output.
        
        Args:
            system: SystemInspector instance for gathering system info
        """
        self.print_section("System Information")

        with self.console.status(
            spinner = "dots",
            status  = "Checking system requirements..."
        ):
            info = system.get_system_info()

        self.console.print(self.create_system_table(info))
        self.console.print()

    def display_wandb(
        self, 
        context : str = "info", 
        project : str = "thermur"
    ) -> str | Literal[False] | None:
        """
        Display wandb information appropriate to the context.
        
        Args:
            context : Display context - "info", "train", or "monitor"
            project : Project name for dashboard links
            
        Returns:
            For monitor context: URL string if ready, False otherwise
            For other contexts: None
        """
        from wandb import Api, api
        
        user = Api().viewer.get("username") if api.api_key  else None
        url  = f"https://wandb.ai/{user}/{project}" if user else None
        
        match context:
            case "info":
                msg = (
                    f"You are logged in as '{user}'" if user 
                    else "🎨 wandb: Not authenticated • Run 'wandb login'"
                )
                self.print_message(
                    message  = msg,
                    msg_type = "flock" if user else "warning"
                )
            case "train" if url:
                self.print_message(f"Live tracking dashboard → {url}", "flock")
            case "monitor":
                if not url:
                    self.print_message(
                        message  = "wandb authentication required to access dashboard",
                        msg_type = "warning"
                    )
                    self.print_message("Run 'wandb login' to authenticate", "info")
                    return False
                self.console.print()
                self.print_message(f"Opening {project} project dashboard...", "flock")
                self.console.print(f"\n  [link={url}]{url}[/link]\n")
                return url
            case _:
                pass

    def format_truncated(
        self,
        text  : str | None = None, 
        value : Any        = None,
        width : int        = 50
    ) -> str:
        """
        Format a string or value with intelligent truncation for display.
        
        Handles different value types (dict, list, string) with appropriate formatting.
        For strings, truncates long values to fit within the specified width while 
        preserving the beginning and end. For dicts with _target_, shows a shortened
        class name. For long lists, shows preview with count.
        
        Args:
            text  : The string to truncate (ignored if value is provided)
            value : Optional value to format (takes precedence over text)
            width : Maximum width for the string (default 50)
            
        Returns:
            Formatted string representation suitable for display
        """
        if value is not None:
            if isinstance(value, str):
                text = value
            else:
                try:
                    text = (
                        f"[{', '.join(map(str, value[:5]))}, ...] ({len(value)} items)" 
                        if len(value) > 5 
                        else str(value)
                    )
                except (TypeError, AttributeError):
                    text = str(value)
        
        if text is None:
            return ""
            
        if len(text) <= width:
            return text
        
        start_chars = width - 13
        return f"{text[:start_chars]}...{text[-10:]}"
    
    def format_summary_list(
        self, 
        items       : list[str], 
        max_display : int = 2,
        separator   : str = ", "
    ) -> str:
        """
        Format a list into a summary string with optional truncation.
        
        Creates a summary like "item1, item2 (+3 more)" for long lists.
        Useful for displaying overrides, file lists, or any collection
        where showing all items would be too verbose.
        
        Args:
            items       : List of items to summarize
            max_display : Maximum number of items to show directly (default 2)
            separator   : String to join items with (default ", ")
            
        Returns:
            Formatted summary string, or "-" if list is empty
            
        Examples:
            >>> format_summary_list(['a', 'b'])
            'a, b'
            >>> format_summary_list(['a', 'b', 'c', 'd', 'e'], max_display=2)
            'a, b (+3)'
        """
        if not items:
            return "-"
            
        if len(items) <= max_display:
            return separator.join(items)
            
        displayed = separator.join(items[:max_display])
        return f"{displayed} (+{len(items) - max_display})"
    
    def format_run_status(self, run_path: Path) -> str:
        """
        Determine and format the status indicator for a training run.
        
        Checks for marker files to determine if a run completed successfully,
        was a dry run, or is incomplete. Returns a Rich-formatted status string.
        
        Args:
            run_path : Path to the run directory
            
        Returns:
            Rich-formatted status indicator: ✓ (complete), ◎ (dry run), or ... (incomplete)
        """
        if (run_path / "training_complete").exists():
            return "[bold green]✓[/]"
        elif (run_path / "dry_run").exists():
            return "[bold cyan]◎[/]"
        else:
            return "[bold yellow]...[/]"
    

    def print_auth_prompt(self, auth_url: str):
        """
        Display authentication prompt with URL.
        
        Shows the authentication URL in a user-friendly format with
        proper styling and instructions.
        
        Args:
            auth_url : The authentication URL to display
        """
        self.console.print()
        self.print_message(
            message  = "Globus authentication required to access dataset files.",
            msg_type = "info"
        )
        self.console.print("\nPlease visit the following URL to authenticate:")
        self.console.print(f"\n  [link={auth_url}]{auth_url}[/link]\n")

    def print_command_example(
        self,
        command     : str,
        description : str,
        note        : str | None = None
    ):
        """
        Print a formatted command example.
        
        Shows command examples in a visually distinct way to help users
        understand how to use the CLI effectively.
        
        Args:
            command     : The actual command to run
            description : What the command does
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

    def print_command_examples(self, examples: list[dict[str, str]]):
        """
        Print multiple command examples from a list.
        
        This method handles the common pattern of iterating through command
        examples and printing each one, eliminating duplication across commands.
        
        Args:
            examples: List of example dictionaries with 'command', 'desc', 
                      and optional 'note' keys
        """
        for example in examples:
            self.print_command_example(
                command     = example["command"],
                description = example["desc"],
                note        = example.get("note", "")
            )

    def print_config_value(
        self,
        key         : str,
        value       : str,
        align_width : int = 0,
        desc        : str | None = None
    ):
        """
        Print a configuration key-value pair.
        
        Formats configuration information in a consistent way for
        improved readability.
        
        Args:
            key         : Configuration key
            value       : Configuration value
            align_width : Width to align the key to (for vertical alignment)
            desc        : Optional description
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

    def print_header(self, title: str):
        """
        Print a styled header panel with fire gradient for "Thermur".
        
        Args:
            title : Main header text
        """
        self.console.print()
        self.console.print(self._create_header_panel(title))
        self.console.print()

    def print_section(
        self,
        title : str,
        minor : bool = False,
        style : str | None = None
    ):
        """
        Print a section divider with appropriate styling.
        
        Creates a visual separator for sections with different prominence
        based on whether it's a major or minor section.
        
        Args:
            title : Section title
            minor : Whether this is a minor section (default: False)
            style : Optional style override (defaults based on major/minor)
        """
        if style is None:
            style = "grey70" if minor else "bright_cyan"
            
        self.console.print()
        self.console.print(
            Rule(
                title      = title if minor else f"[bold]{title}[/bold]", 
                style      = style, 
                characters = "─" if minor else "═", 
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
