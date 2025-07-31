"""
Manage and explore training runs and their configurations.

This module provides the 'runs' command for interacting with Hydra's output
directories. It enables users to list training runs, compare configurations 
between runs, clean up old experiments, and inspect detailed configuration 
settings with pagination support for large configurations.
"""
from collections import ChainMap
from contextlib  import contextmanager, suppress
from itertools   import chain
from pathlib     import Path
from shutil      import rmtree
from thermur.cli import app
from typer       import Argument, Context, Exit, Option, Typer
from typing      import Any, Iterator, Optional
from yaml        import Loader, load, safe_load

runs = Typer(
    help             = "🏃 Explore training runs and configurations",
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
    
    When called without a subcommand, shows available commands.
    Use subcommands to view specific configurations or manage outputs.
    """
    if ctx.invoked_subcommand is None:
        app.ui.print_header("Training Run Management")
        ctx.get_help()


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
    Clean up old training runs. Use --keep N to preserve recent runs.
    
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
    run1   : Optional[str] = Argument(
        None, 
        help="First run ID or path. Use 'last[N]' for Nth most recent (defaults to 'last[1]')"
    ),
    run2   : Optional[str] = Argument(
        None, 
        help="Second run ID or path. Use 'last[N]' for Nth most recent (defaults to 'last[2]')"
    ),
    domain : Optional[str] = Option(
        None,
        "--domain", "-d",
        help = "Compare only specific domain (e.g., controller, lightning)"
    )
):
    """
    Compare configurations between two runs. Defaults to last 2.
    
    Displays side-by-side differences between the configurations of two
    training runs. You can filter the comparison to a specific domain
    (e.g., controller, lightning) to focus on relevant settings.
    
    When no arguments are provided, compares the two most recent runs.
    You can use 'last[N]' syntax to reference the Nth most recent run.
    
    Examples:
        thermur runs compare                                   # Compare last 2 runs
        thermur runs compare last outputs/2025-07-29/15-30-00  # Compare most recent to specific
        thermur runs compare last[1] last[3]                   # Compare most recent to 3rd most recent
        thermur runs compare run1 run2 -d lightning            # Compare specific domain
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
    List training runs with timestamps and status indicators.
    
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
        help = (
            "Run ID or path. Use 'last' for most recent, 'last[N]' for Nth "
            "most recent (defaults to 'last')"
        )
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
    Display run configuration. Use 'last[N]' for Nth most recent.
    
    Shows the full Hydra configuration for a training run, optionally filtered
    by domain. Large configurations are automatically paginated for better 
    readability. System configurations (prefixed with _) are hidden by default
    but can be included with the --all flag.
    
    Examples:
        thermur runs show                                 # Show last run
        thermur runs show last                            # Explicitly show last  
        thermur runs show last[2]                         # Show 2nd most recent run
        thermur runs show outputs/2025-07-30/10-15-30     # Show a specific run
        thermur runs show -d controller                   # Show only controller config
        thermur runs show --all                           # Include system (_) configs
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

    def _display_config(
        self,
        items      : list[tuple[str, Any]],
        title      : str,
        run_path   : Optional[Path] = None,
        overrides  : Optional[list[str]] = None,
        page_size  : int = 20,
        using_last : bool = False
    ):
        """
        Unified config display using pagination for all cases.
        
        Args:
            items      : List of (path, (value, is_override)) tuples
            title      : Title for the configuration section
            run_path   : Optional path to the run for display context
            overrides  : List of override strings from Hydra
            page_size  : Number of items per page (default 20)
            using_last : Whether the user is viewing the last run
        """
        sorted_items = sorted(items, key=lambda x: x[0])
        
        def render_page(page_items, page_num, total_pages):
            self.ui.print_header("Training Run Management")
            
            if run_path:
                self.ui.print_section(
                    f"Configuration: {run_path.relative_to(self.outputs_dir)}",
                    minor=True
                )
            
            if page_num == 1:
                if using_last:
                    self.ui.print_message(
                        message  = (
                            "Using most recent run. "
                            "(Use specific run ID to view others)"
                        ),
                        msg_type = "info"
                    )
                
                if overrides:
                    self._display_overrides_panel(overrides)
            
            page_info = (
                f" (Page {page_num}/{total_pages})" if total_pages > 1 else ""
            )
            self.ui.print_section(
                style = "bright_cyan",
                title = f"{title}{page_info}"
            )
            
            table = self.ui.create_aligned_table(
                border_style = "dim",
                columns      = [
                    ("",          "bright_yellow", 2,  "center"),
                    ("Parameter", "bright_white",  48, "left"),
                    ("Value",     "bright_green",  50, "left")
                ]
            )
            
            self._add_config_rows(table, page_items)
            self.ui.display_panel(table)
        
        self.prompts.paginate(
            allow_row_select = False,
            items            = sorted_items,
            page_size        = page_size,
            render_page      = render_page
        )
    
    
    def _display_overrides_panel(self, overrides: list):
        """
        Display overrides in a formatted panel.
        
        Creates a warning panel showing all configuration overrides applied
        to a training run, formatted as a bulleted list.
        
        Args:
            overrides : List of override strings from Hydra
        """
        issues = overrides + ["Look for ● markers in the configuration below"]
        panel = self.ui.create_warning_panel(
            issues = issues,
            title  = "Configuration Overrides"
        )
        self.ui.display_panel(panel)
    
    def _display_runs_table(
        self,
        runs         : list[Path],
        columns      : list[tuple[str, str, int, str]],
        border_style : str = "bright_blue",
        title        : Optional[str] = None,
        show_overrides : bool = True
    ) -> None:
        """
        Display runs in a formatted table with status indicators.
        
        Args:
            runs         : List of run paths to display
            columns      : Column definitions for the table
            border_style : Border style for the table
            title        : Optional title for the table
            show_overrides : Whether to include overrides column
        """
        table = self.ui.create_aligned_table(
            border_style = border_style,
            columns      = columns,
            title        = title
        )
        
        for run_path in runs:
            run_id = str(run_path.relative_to(self.outputs_dir))
            
            if (run_path / "training_complete").exists():
                status = "[bold green]✓[/]"
            elif (run_path / "dry_run").exists():
                status = "[bold cyan]◎[/]"
            else:
                status = "[bold yellow]...[/]"
            
            if show_overrides:
                overrides = (
                    self._format_overrides(run_path / ".hydra" / "overrides.yaml")
                    if (run_path / ".hydra" / "overrides.yaml").exists()
                    else "-"
                )
                table.add_row(run_id, overrides, status)
            else:
                table.add_row(run_id, status)
        
        self.ui.display_panel(table)
    
    
    def _flatten_config(
        self,
        config    : dict[str, Any],
        overrides : Optional[list[str]] = None,
        prefix    : str = ""
    ) -> dict[str, Any]:
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
    
    def _load_run_config(
        self,
        run_path       : Path,
        include_system : bool = False
    ) -> dict[str, Any]:
        """
        Load and optionally filter a run's configuration.
        
        Args:
            run_path       : Path to the run directory
            include_system : Whether to include system (_) configs
            
        Returns:
            Configuration dictionary
        """
        with load_yaml(run_path / ".hydra" / "config.yaml") as cfg:
            if not include_system:
                return {k: v for k, v in cfg.items() if not k.startswith('_')}
            return cfg
    
    def _get_domains(
        self, 
        cfg: dict[str, Any], 
        domain: Optional[str] = None
    ) -> list[str]:
        """
        Get list of domains to process from configuration.
        
        Args:
            cfg    : Configuration dictionary
            domain : Optional specific domain to filter to
            
        Returns:
            List of domain names to process
        """
        if domain:
            if domain not in cfg:
                self.ui.print_message(
                    message  = f"Domain '{domain}' not found in configuration",
                    msg_type = "warning"
                )
                return []
            return [domain]
        return sorted(cfg.keys())
    
    def _load_overrides(self, run_path: Path) -> list[str]:
        """
        Load override configuration from a run.
        
        Args:
            run_path : Path to the run directory
            
        Returns:
            List of override strings, or empty list if none found
        """
        overrides_file = run_path / ".hydra" / "overrides.yaml"
        if overrides_file.exists():
            with suppress(Exception), open(overrides_file) as f:
                return safe_load(f) or []
        return []
    
    
    def _resolve_run_id(self, run_id: str) -> Path:
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
        
        self._display_runs_table(
            runs           = to_delete,
            columns        = columns,
            border_style   = "red",
            title          = (
                f"{len(to_delete)} run{'s' if len(to_delete) > 1 else ''} "
                f"will be deleted"
            ),
            show_overrides = False
        )
        
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
        
        cfg1 = self._load_run_config(run1_path)
        cfg2 = self._load_run_config(run2_path)
        
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
        domains     = self._get_domains(dict(all_configs), domain)
        
        # Collect all domains with differences
        domains_with_diffs = []
        
        for domain in (d for d in domains if d in all_configs):
            flat1 = self._flatten_config(cfg1.get(domain, {}))
            flat2 = self._flatten_config(cfg2.get(domain, {}))
            
            all_keys    = set(flat1) | set(flat2)
            differences = []
            for key in sorted(all_keys):
                val1 = flat1.get(key, ("NOT SET", False))[0]
                val2 = flat2.get(key, ("NOT SET", False))[0]
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
        
        if limit and len(runs) > limit:
            self.ui.print_message(
                message  = (
                    f"Showing {limit} of {len(runs)} runs. "
                    f"(Use --all to see all runs)"
                ),
                msg_type = "info"
            )
        elif limit and len(runs) <= limit:
            self.ui.print_message(
                message  = f"Showing all {len(runs)} runs.",
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
        
        self._display_runs_table(
            runs           = display_runs,
            columns        = columns,
            border_style   = "bright_blue",
            show_overrides = True
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
        run_path  = self._resolve_run_id(run_id)
        cfg       = self._load_run_config(run_path, include_system)
        overrides = self._load_overrides(run_path)
        
        domains = self._get_domains(cfg, domain)
        if not domains:
            return
        
        all_items = []
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
        
        self._display_config(
            items      = all_items,
            title      = "Configuration Parameters",
            run_path   = run_path,
            overrides  = overrides,
            using_last = run_id == "last"
        )
