"""
Manage and explore training runs and their configurations.

This module provides the 'runs' command for interacting with Hydra's output
directories. It enables users to list training runs, compare configurations 
between runs, clean up old experiments, and inspect detailed configuration 
settings with pagination support for large configurations.
"""
from collections import ChainMap
from contextlib  import ExitStack, contextmanager, suppress
from itertools   import chain
from pathlib     import Path
from shutil      import rmtree
from thermur.cli import app
from typer       import Argument, Context, Exit, Option, Typer
from typing      import Any, Iterator, Optional, TypeAlias
from yaml        import Loader, load, safe_load

ConfigDict : TypeAlias = dict[str, Any]
RunPath    : TypeAlias = Path

runs = Typer(
    help             = "Explore and manage training runs",
    rich_markup_mode = "rich",
)


@contextmanager
def load_yaml(path: Path) -> Iterator[Any]:
    """
    Context manager for loading YAML files with error handling.
    
    Provides a clean interface for YAML file operations with automatic
    resource management and consistent error handling.
    
    Args:
        path: Path to the YAML file
        
    Yields:
        Loaded YAML content
        
    Raises:
        Exit: If file cannot be loaded
    """
    try:
        with open(path) as f:
            yield load(f, Loader)
            
    except Exception as e:
        app.ui.print_message(
            message  = f"Failed to load {path}: {e}",
            msg_type = "error"
        )
        raise Exit(1)


@runs.callback(invoke_without_command=True)
def runs_callback(ctx: Context):
    """
    🏃 Explore training runs and configurations.
    
    When called without a subcommand, lists recent training runs.
    Use subcommands to view specific configurations or manage outputs.
    """
    # Only run the list if no subcommand is provided
    if ctx.invoked_subcommand is None:
        RunsCommand().list_runs(None, show_header=True)


@runs.command("clean")
def clean(
    dry_run: bool = Option(
        False,
        "--dry-run", "-d",
        help = "Show what would be deleted without deleting"
    ),
    force: bool = Option(
        False,
        "--force", "-f",
        help = "Skip confirmation prompt"
    ),
    keep: int = Option(
        0,
        "--keep", "-k",
        help = "Number of recent runs to keep"
    )
):
    """
    Clean up old training runs.
    
    This command removes training runs from the outputs directory. By default,
    it deletes ALL runs. Use the --keep option to preserve recent runs.
    Use dry-run mode to preview what would be deleted.
    
    Examples:
        thermur runs clean              # Delete all runs
        thermur runs clean -k 5         # Delete all but 5 most recent
        thermur runs clean --dry-run    # Preview what would be deleted
    """
    RunsCommand().clean_runs(dry_run, force, keep)


@runs.command("compare")
def compare(
    run1   : Optional[str] = Argument(None, help="First run ID or path (defaults to 'last[1]')"),
    run2   : Optional[str] = Argument(None, help="Second run ID or path (defaults to 'last[2]')"),
    domain : Optional[str] = Option(
        None,
        "--domain", "-d",
        help = "Compare only specific domain"
    )
):
    """
    Compare configurations between two training runs.
    
    This command displays side-by-side differences between the configurations
    of two training runs. You can filter the comparison to a specific domain
    (e.g., controller, lightning) to focus on relevant settings.
    
    When no arguments are provided, compares the two most recent runs.
    You can use 'last[N]' syntax to reference the Nth most recent run.
    
    Args:
        run1   : First run identifier (path, "last", or "last[N]" for Nth most recent)
        run2   : Second run identifier (path, "last", or "last[N]" for Nth most recent)
        domain : Optional domain to filter comparison (e.g., controller)
        
    Examples with `thermur runs ...`:
        compare                                   # Compare last 2 runs
        compare last outputs/2025-07-29/15-30-00  # Compare most recent to specific
        compare last[1] last[3]                   # Compare most recent to 3rd most recent
        compare run1 run2 -d lightning            # Compare specific domain
    """
    if run1 is None and run2 is None:
        run1 = "last[1]"
        run2 = "last[2]"
    elif run1 is not None and run2 is None:
        run2 = "last[1]"
    
    RunsCommand().compare_runs(domain, run1, run2)


@runs.command("list")
def list_command(
    all_runs: bool = Option(
        False,
        "--all", "-a", 
        help = "Show all runs (ignore limit)"
    ),
    limit: int = Option(
        10,
        "--limit", "-n",
        help = "Number of recent runs to display"
    )
):
    """
    List recent training runs with timestamps and status.
    
    Displays a formatted table of training runs including their IDs, timestamps,
    configuration overrides, and completion status. By default shows the 10 most
    recent runs, but this can be adjusted with the limit option.
    
    Examples:
        thermur runs list           # Show 10 most recent
        thermur runs list -n 20     # Show 20 most recent
        thermur runs list --all     # Show all runs
    """
    RunsCommand().list_runs(None if all_runs else limit, show_header=True)


@runs.command("show")
def show(
    run_id: Optional[str] = Argument(
        None,
        help = "Run ID or path (defaults to 'last')"
    ),
    all_configs: bool = Option(
        False,
        "--all", "-a",
        help = "Include system (_) configurations"
    ),
    domain: Optional[str] = Option(
        None,
        "--domain", "-d",
        help = "Show only specific domain (e.g., controller, lightning)"
    )
):
    """
    Display configuration for a specific training run.
    
    Shows the full Hydra configuration for a training run, optionally filtered
    by domain. Large configurations are automatically paginated for better 
    readability. System configurations (prefixed with _) are hidden by default
    but can be included with the --all flag.
    
    Examples with `thermur runs ...`:
        show                             # Show last run
        show last                        # Explicitly show last  
        show last[2]                     # Show 2nd most recent run
        show outputs/2025-07-30/10-15-30 # See a specific run
        show -d controller               # Show only controller config
        show --all                       # Include _system configs
    """
    RunsCommand().show_config(
        domain         = domain,
        include_system = all_configs,
        run_id         = run_id or "last"
    )


class RunsCommand:
    """
    Encapsulates the state and logic for the 'runs' command.
    
    This class provides methods for listing, comparing, cleaning, and displaying
    training run configurations from Hydra's output directories. Private methods
    handle internal operations like path resolution and formatting.
    """
    
    def __init__(self):
        """
        Initializes the command with shared context components.
        
        Sets up references to the configuration, UI components, and other
        shared application resources needed for run management operations.
        """
        self.cfg         = app.cfg
        self.outputs_dir = Path("outputs")
        self.prompts     = app.prompts
        self.system      = app.system
        self.ui          = app.ui
    
    def _add_config_rows(self, table, items):
        """
        Add configuration rows to table with override indicators.
        """
        for path, (value, is_override) in items:
            indicator = "●" if is_override else " "
            table.add_row(indicator, path, self._format_value(value))

    def _display_all_config(
        self,
        items      : list[tuple[str, Any]],
        run_path   : Optional[Path] = None,
        overrides  : Optional[list[str]] = None,
        using_last : bool = False
    ):
        """
        Display all configuration parameters in a single paginated view.
        
        Args:
            items     : List of (path, value) tuples to display
            run_path  : Optional path to the run for display context
            overrides : List of override strings from Hydra
        """
        columns = [
            ("",          "bright_yellow", 2,  "center"),
            ("Parameter", "bright_white",  48, "left"),
            ("Value",     "bright_green",  50, "left")
        ]
        page_size    = 20
        sorted_items = sorted(items, key=lambda x: x[0])
        
        if len(sorted_items) <= page_size:
            if using_last:
                self.ui.print_message(
                    message  = (
                        "Using most recent run. "
                        "(Use specific run ID to view others)"
                    ),
                    msg_type = "info"
                )
            
            self.ui.print_header("Training Run Management")
            self.ui.print_section(
                f"Configuration: {run_path.relative_to(self.outputs_dir)}",
                minor=True
            )
            
            if overrides:
                self._display_overrides_panel(overrides)
            
            self.ui.print_section(
                style = "bright_cyan",
                title = "Configuration Parameters"
            )
            
            table = self.ui.create_aligned_table(
                border_style = "dim",
                columns      = columns
            )
            
            for path, (value, is_override) in sorted_items:
                indicator = "●" if is_override else " "
                table.add_row(indicator, path, self._format_value(value))
            
            self.ui.display_panel(table)
            self.ui.print_message(
                message  = "● = Parameter overridden from default",
                msg_type = "info"
            )
        else:
            self._paginate_all_config(
                columns    = columns, 
                items      = sorted_items, 
                overrides  = overrides,
                page_size  = page_size, 
                run_path   = run_path,
                using_last = using_last
            )
    
    def _display_domain_config(
        self,
        config    : ConfigDict,
        domain    : str,
        run_path  : Optional[Path] = None,
        overrides : Optional[list[str]] = None
    ):
        """
        Display configuration for a single domain with pagination.
        
        Flattens nested configuration dictionaries into dot-notation paths
        and displays them in a paginated table format. Shows override
        indicators for modified values.
        
        Args:
            config    : The configuration dictionary to display
            domain    : The configuration domain name (e.g., 'controller')
            run_path  : Optional path to the run for display context
            overrides : List of override strings from Hydra
        """
        if not (items := list(self._flatten_config(config, overrides=overrides).items())):
            return
        
        columns = [
            ("", "bright_yellow", 2, "center"),
            ("Parameter", "bright_white", 48, "left"),
            ("Value",     "bright_green", 50, "left")
        ]
        page_size = 20
        
        if len(items) <= page_size:
            self.ui.print_section(
                style = "bright_cyan",
                title = f"{domain.title()} Configuration"
            )
            
            table = self.ui.create_aligned_table(
                border_style = "dim",
                columns      = columns
            )
            
            for path, (value, is_override) in items:
                indicator = "●" if is_override else " "
                table.add_row(indicator, path, self._format_value(value))
            
            self.ui.display_panel(table)
            self.ui.print_message(
                message  = "● = Parameter overridden from default",
                msg_type = "info"
            )
        else:
            self._paginate_config(columns, domain, items, page_size, run_path)
    
    def _display_overrides_panel(self, overrides: list):
        """
        Display overrides in a formatted panel.
        
        Creates a warning panel showing all configuration overrides applied
        to a training run, formatted as a bulleted list.
        
        Args:
            overrides : List of override strings from Hydra
        """
        panel = self.ui.create_warning_panel(
            issues = overrides,
            title  = "Configuration Overrides"
        )
        self.ui.display_panel(panel)
    
    def _flatten_config(
        self,
        config    : ConfigDict,
        overrides : Optional[list[str]] = None,
        prefix    : str = ""
    ) -> ConfigDict:
        """
        Flatten nested config to dot notation.
        
        Recursively flattens a nested configuration dictionary into a flat
        dictionary with dot-notation keys. Expands model instances to show
        their parameters. Tracks which values were overridden.
        
        Args:
            config    : Configuration dictionary to flatten
            overrides : List of override paths from Hydra
            prefix    : Current path prefix for recursion
            
        Returns:
            Flattened dictionary with dot-notation keys and override info
        """
        items = {}
        override_set = set()
        
        if overrides:
            override_set = {
                override.split('=')[0].strip('+') 
                for override in overrides 
                if '=' in override
            }
        
        for key, value in config.items():
            if key.startswith('_') and key != '_target_':
                continue
            
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                if '_target_' in value:
                    model_params = {
                        k: v for k, v in value.items() 
                        if k != '_target_'
                    }
                    if not model_params:
                        items[full_key] = (value['_target_'].split('.')[-1], False)
                        continue
                    value = model_params
                
                items.update(self._flatten_config(
                    config    = value,
                    overrides = overrides,
                    prefix    = full_key
                ))
            else:
                is_override = full_key in override_set or any(
                    full_key.startswith(p) for p in override_set
                )
                items[full_key] = (value, is_override)
        
        return items
    
    def _format_overrides(self, overrides_file: Path) -> str:
        """
        Format overrides from file for display.
        
        Reads overrides from a YAML file and formats them for compact display
        in a table. Shows up to two overrides directly and indicates if there
        are more with a count.
        
        Args:
            overrides_file : Path to the overrides.yaml file
            
        Returns:
            Formatted string representation of overrides
        """
        with suppress(Exception):
            with open(overrides_file) as f:
                if overrides := safe_load(f) or []:
                    return (
                        ", ".join(overrides[:2]) + f" (+{len(overrides)-2})"
                        if len(overrides) > 2
                        else ", ".join(overrides)
                    )
        return "-"
    
    def _format_value(self, value: Any) -> str:
        """
        Format a configuration value for display.
        
        Handles different value types (dict, list, string) and formats them
        appropriately for table display. Uses the UI helper for consistent
        truncation across the application.
        
        Args:
            value : Value to format
            width : Maximum width for display (default 50 for table columns)
            
        Returns:
            Formatted string representation suitable for display
        """
        match value:
            case dict() if '_target_' in value:
                return f"{value['_target_'].split('.')[-1]}(...)"
            case list() if len(value) > 5:
                preview = ', '.join(str(v) for v in value[:5])
                return f"[{preview}, ...] ({len(value)} items)"
            case _:
                return self.ui.format_truncated(str(value))
    
    def _get_all_runs(self) -> list:
        """
        Get all run directories sorted by timestamp.
        
        Scans the outputs directory for Hydra run folders (identified by the
        presence of a .hydra subdirectory) and returns them in reverse
        chronological order. This ensures the most recent runs appear first.
        
        Returns:
            List of Path objects pointing to valid run directories
        """
        if not self.outputs_dir.exists():
            return []
        
        all_paths = chain.from_iterable(
            sorted(date_dir.iterdir(), reverse=True)
            for date_dir in sorted(self.outputs_dir.iterdir(), reverse=True)
            if date_dir.is_dir()
        )
        
        return [
            path for path in all_paths
            if path.is_dir() and (path / ".hydra").exists()
        ]
    
    def _paginate_all_config(
        self,
        columns   : list,
        items     : list[tuple[str, Any]],
        page_size : int,
        run_path  : Optional[Path] = None,
        overrides : Optional[list[str]] = None
    ):
        """
        Handle paginated display of all configuration parameters.
        
        Args:
            columns   : Column definitions for the table
            items     : Sorted list of (path, value) tuples
            page_size : Number of items to show per page
            run_path  : Optional path to the run for display context
        """
        def render_config_page(
            page_items  : list, 
            page_num    : int, 
            total_pages : int
        ):
            """
            Render a page of configuration parameters in a table.
            """
            self.ui.print_header("Training Run Management")
            if run_path:
                self.ui.print_section(
                    f"Configuration: {run_path.relative_to(self.outputs_dir)}",
                    minor=True
                )
            
            if page_num == 1 and overrides:
                self._display_overrides_panel(overrides)
            
            self.ui.print_section(
                style = "bright_cyan",
                title = f"Configuration Parameters (Page {page_num}/{total_pages})"
            )
            
            table = self.ui.create_aligned_table(
                border_style = "dim",
                columns      = columns
            )
            
            for path, (value, is_override) in page_items:
                indicator = "●" if is_override else " "
                table.add_row(indicator, path, self._format_value(value))
            
            self.ui.display_panel(table)
            self.ui.print_message(
                message  = "● = Parameter overridden from default",
                msg_type = "info"
            )
        
        self.prompts.paginate(
            allow_row_select = False,
            items            = items,
            page_size        = page_size,
            render_page      = render_config_page
        )
    
    def _paginate_config(
        self,
        columns   : list,
        domain    : str,
        items     : list,
        page_size : int,
        run_path  : Optional[Path] = None
    ):
        """
        Handle paginated display of large configurations.
        
        Uses the prompts.paginate method to display configuration items
        across multiple pages with navigation controls.
        
        Args:
            columns   : Column definitions for the table
            domain    : Configuration domain name for the title
            items     : List of (key, value) tuples to display
            page_size : Number of items to show per page
            run_path  : Optional path to the run for display context
        """
        def render_config_page(
            page_items  : list, 
            page_num    : int, 
            total_pages : int
        ):
            """
            Render a page of configuration parameters in a table.
            """
            self.ui.print_header("Training Run Management")
            if run_path:
                self.ui.print_section(
                    f"Configuration: {run_path.relative_to(self.outputs_dir)}",
                    minor=True
                )
            
            self.ui.print_section(
                style = "bright_cyan",
                title = (
                    f"{domain.title()} Configuration "
                    f"(Page {page_num}/{total_pages})"
                )
            )
            
            table = self.ui.create_aligned_table(
                border_style = "dim",
                columns      = columns
            )
            
            for path, (value, is_override) in page_items:
                indicator = "●" if is_override else " "
                table.add_row(indicator, path, self._format_value(value))
            
            self.ui.display_panel(table)
        
        self.prompts.paginate(
            allow_row_select = False,
            items            = items,
            page_size        = page_size,
            render_page      = render_config_page
        )
    
    def _resolve_run_id(self, run_id: str) -> RunPath:
        """
        Resolve run ID to actual path.
        
        Handles special "last" alias for the most recent run, "last[N]" for the
        Nth most recent run, or converts the provided run ID string to a Path
        object and validates it exists. This provides a consistent way to
        reference runs by various identifiers.
        
        Args:
            run_id : Either "last", "last[N]", or a path to a run directory
            
        Returns:
            Path object pointing to the resolved run directory
            
        Raises:
            Exit : If no runs exist or specified run is not found
        """
        if run_id.startswith("last"):
            if not (runs := self._get_all_runs()):
                self.ui.print_message(
                    message  = "No training runs found in outputs/",
                    msg_type = "error"
                )
                raise Exit(1)
            
            if run_id == "last":
                n = 1
            elif run_id.startswith("last[") and run_id.endswith("]"):
                try:
                    n = int(run_id[5:-1])
                    if n < 1:
                        raise ValueError("N must be >= 1")
                except ValueError:
                    self.ui.print_message(
                        message  = (
                            f"Invalid last[N] syntax: {run_id}. "
                            f"N must be a positive integer."
                        ),
                        msg_type = "error"
                    )
                    raise Exit(1)
            else:
                self.ui.print_message(
                    message  = (
                        f"Invalid run identifier: {run_id}. "
                        f"Use 'last', 'last[N]', or a valid path."
                    ),
                    msg_type = "error"
                )
                raise Exit(1)
            
            if n > len(runs):
                self.ui.print_message(
                    message  = f"Only {len(runs)} runs found, cannot get run #{n}",
                    msg_type = "error"
                )
                raise Exit(1)
            
            return runs[n - 1]
        
        if not (run_path := Path(run_id)).exists():
            self.ui.print_message(
                message  = f"Run not found: {run_id}",
                msg_type = "error"
            )
            raise Exit(1)
        
        return run_path
    
    def clean_runs(
        self,
        dry_run : bool,
        force   : bool,
        keep    : int
    ):
        """
        Clean up old training runs.
        
        Removes old training runs from the outputs directory, keeping only
        the specified number of most recent runs. Provides safety features
        including dry-run mode and confirmation prompts.
        
        Args:
            dry_run : Whether to simulate deletion without actually deleting
            force   : Whether to skip confirmation prompt
            keep    : Number of recent runs to keep
        """
        self.ui.print_header("Training Run Management")
        
        runs = self._get_all_runs()
        
        if len(runs) <= keep:
            self.ui.print_message(
                message  = f"Only {len(runs)} runs found. Nothing to clean.",
                msg_type = "info"
            )
            return
        
        to_delete = runs[keep:]
        
        self.ui.print_section("Runs to Delete", minor=True)
        
        columns = [
            ("Run ID",      "bright_red",    50, "left"),
            ("Status",      "bright_white",  10, "center")
        ]
        
        table = self.ui.create_aligned_table(
            border_style = "red",
            columns      = columns,
            title        = (
                f"{len(to_delete)} run{'s' if len(to_delete) > 1 else ''} "
                f"will be deleted"
            )
        )
        
        for run_path in to_delete:
            run_id = str(run_path.relative_to(self.outputs_dir))
            
            if (run_path / "training_complete").exists():
                status = "[bold green]✓[/]"
            elif (run_path / "dry_run").exists():
                status = "[bold cyan]◎[/]"
            else:
                status = "[bold yellow]...[/]"
            
            table.add_row(run_id, status)
        
        self.ui.display_panel(table)
        
        if dry_run:
            self.ui.print_message(
                message  = f"Would delete {len(to_delete)} runs (dry run mode)",
                msg_type = "info"
            )
            return
        
        issues = [
            (
                f"This will permanently delete {len(to_delete)} training "
                f"run{'s' if len(to_delete) > 1 else ''}"
            ),
            "This action cannot be undone"
        ]
        
        if keep > 0:
            issues.append(f"Keeping only the {keep} most recent runs")
        else:
            issues.append("Use --keep N to preserve N recent runs")
            
        warning_panel = self.ui.create_warning_panel(
            issues = issues,
            title  = "⚠️  Confirm Deletion"
        )
        self.ui.display_panel(warning_panel)
        
        if not force and not self.prompts.confirm(
            f"Delete {len(to_delete)} old runs?"
        ):
            self.ui.print_message(
                message  = "Cleanup cancelled",
                msg_type = "warning"
            )
            return
        
        progress = self.ui.create_thermal_progress()
        task     = progress.add_task("Deleting runs...", total=len(to_delete))
        deleted  = 0
        
        with progress:
            for run in to_delete:
                with suppress(Exception):
                    rmtree(run)
                    deleted += 1
                    progress.update(advance=1, task=task)
        
        self.ui.print_message(
            message  = f"Deleted {deleted} old runs",
            msg_type = "success"
        )
    
    def compare_runs(
        self,
        domain : Optional[str],
        run1   : str,
        run2   : str
    ):
        """
        Compare configurations between two runs.
        
        Displays side-by-side differences between the configurations of two
        training runs. Can be filtered to show only a specific domain.
        
        Args:
            domain : Optional domain to filter comparison
            run1   : First run identifier
            run2   : Second run identifier
        """
        if run1 == "last[1]" and run2 == "last[2]":
            self.ui.print_message(
                message  = "Comparing the two most recent runs.",
                msg_type = "info"
            )
        
        run1_path = self._resolve_run_id(run1)
        run2_path = self._resolve_run_id(run2)
        
        with ExitStack() as stack:
            cfg1 = stack.enter_context(
                load_yaml(run1_path / ".hydra" / "config.yaml")
            )
            cfg2 = stack.enter_context(
                load_yaml(run2_path / ".hydra" / "config.yaml")
            )
            
            cfg1 = {k: v for k, v in cfg1.items() if not k.startswith('_')}
            cfg2 = {k: v for k, v in cfg2.items() if not k.startswith('_')}
        
        self.ui.print_header("Training Run Management")
        self.ui.print_section("Configuration Comparison", minor=True)
        self.ui.print_message(
            message  = f"Run 1: {run1_path.relative_to(self.outputs_dir)}",
            msg_type = "info"
        )
        self.ui.print_message(
            message  = f"Run 2: {run2_path.relative_to(self.outputs_dir)}",
            msg_type = "info"
        )
        
        all_configs = ChainMap(cfg1, cfg2)
        domains     = [domain] if domain else sorted(set(all_configs.keys()))
        
        # Collect all domains with differences
        domains_with_diffs = []
        
        for domain in (d for d in domains if d in all_configs):
            flat1 = self._flatten_config(cfg1.get(domain, {}))
            flat2 = self._flatten_config(cfg2.get(domain, {}))
            
            def get_value(flat_dict, key, default="NOT SET"):
                item = flat_dict.get(key, default)
                if isinstance(item, tuple):
                    return item[0]
                return item
            
            all_keys    = set(flat1) | set(flat2)
            differences = []
            for key in sorted(all_keys):
                val1 = get_value(flat1, key)
                val2 = get_value(flat2, key)
                if val1 != val2:
                    differences.append((key, val1, val2))
            
            if differences:
                domains_with_diffs.append((domain, differences))
        
        if not domains_with_diffs:
            self.ui.print_message(
                message  = "No configuration differences found between runs",
                msg_type = "info"
            )
            return
        
        # Display differences for each domain
        for domain, differences in domains_with_diffs:
            table = self.ui.create_aligned_table(
                border_style = "dim",
                columns      = [
                    ("Parameter", "bright_white", 40, "left"),
                    ("Run 1",     "bright_green", 30, "left"),
                    ("Run 2",     "bright_blue",  30, "left")
                ],
                title        = f"{domain.title()} Configuration"
            )
            
            for param, val1, val2 in differences:
                table.add_row(
                    param,
                    self._format_value(val1),
                    self._format_value(val2)
                )
            
            self.ui.display_panel(table)
    
    def list_runs(self, limit: Optional[int] = None, show_header: bool = True):
        """
        List recent training runs.
        
        Displays a formatted table of training runs including their IDs,
        timestamps, configuration overrides, and completion status.
        
        Args:
            limit       : Maximum number of runs to display (None for all)
            show_header : Whether to show the section header
        """
        if show_header:
            self.ui.print_header("Training Run Management")
            
        runs = self._get_all_runs()
        if not runs:
            self.ui.print_message(
                message  = "No training runs found. Start training with: "  
                           "thermur train",
                msg_type = "info"
            )
            return
        
        display_runs = runs if limit is None else runs[:limit]
        
        if limit is not None and limit == 10:
            if len(display_runs) < limit:
                self.ui.print_message(
                    message  = f"Showing all {len(display_runs)} runs.",
                    msg_type = "info"
                )
            else:
                self.ui.print_message(
                    message  = (
                        f"Showing {len(display_runs)} most recent runs. "
                        f"(Use -n to change limit or --all)"
                    ),
                    msg_type = "info"
                )
        
        self.ui.print_section("Recent Runs", minor=True)
        self.ui.console.print(
            "ℹ️  Status: "
            "[bold green](✓) Complete[/], "
            "[bold cyan](◎) Dry Run[/], "
            "[bold yellow](...) Incomplete[/]"
        )
        
        columns = [
            ("Run ID",       "bright_cyan",   35, "left"),
            ("Overrides",    "bright_green",  55, "left"),
            ("Status",       "bright_white",  10, "center")
        ]
        
        table = self.ui.create_aligned_table(
            border_style = "bright_blue",
            columns      = columns
        )
        
        for run_path in display_runs:
            run_id = str(run_path.relative_to(self.outputs_dir))
            
            overrides = (
                self._format_overrides(run_path / ".hydra" / "overrides.yaml")
                if (run_path / ".hydra" / "overrides.yaml").exists()
                else "-"
            )
            
            if (run_path / "training_complete").exists():
                status = "[bold green]✓[/]"
            elif (run_path / "dry_run").exists():
                status = "[bold cyan]◎[/]"
            else:
                status = "[bold yellow]...[/]"
            
            table.add_row(run_id, overrides, status)
        
        self.ui.display_panel(table)
        
        if limit and len(runs) > limit and limit != 10:
            self.ui.print_message(
                message  = (
                    f"Showing {limit} of {len(runs)} runs. "
                    f"(Use --all to see all runs)"
                ),
                msg_type = "info"
            )
    
    def show_config(
        self,
        domain         : Optional[str],
        include_system : bool,
        run_id         : str
    ):
        """
        Display configuration for a run with pagination.
        
        Shows the full Hydra configuration for a training run, optionally
        filtered by domain. Large configurations are automatically paginated
        for better readability.
        
        Args:
            domain         : Optional specific domain to display
            include_system : Whether to include system (_) configurations
            run_id         : Run identifier or "last" for most recent
        """
        run_path = self._resolve_run_id(run_id)
        
        with load_yaml(run_path / ".hydra" / "config.yaml") as cfg:
            if not include_system:
                cfg = {k: v for k, v in cfg.items() if not k.startswith('_')}
        
        overrides = []
        if (overrides_file := run_path / ".hydra" / "overrides.yaml").exists():
            with suppress(Exception), open(overrides_file) as f:
                if loaded_overrides := safe_load(f) or []:
                    overrides = loaded_overrides
        
        all_items = []
        
        if domain:
            if domain not in cfg:
                self.ui.print_message(
                    message  = f"Domain '{domain}' not found in configuration",
                    msg_type = "warning"
                )
                return
            domains = [domain]
        else:
            domains = sorted(cfg.keys())
        
        for d in domains:
            domain_config = self._flatten_config(
                config    = cfg[d], 
                overrides = overrides,
                prefix    = d
            )
            all_items.extend(domain_config.items())
        
        if not all_items:
            self.ui.print_message(
                message  = "No configuration parameters found",
                msg_type = "info"
            )
            return
        
        self._display_all_config(all_items, run_path, overrides, run_id == "last")
