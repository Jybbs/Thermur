"""
Manage and explore training runs and their configurations.

This module provides the 'runs' command for interacting with Hydra's output
directories. It enables users to list training runs, compare configurations 
between runs, clean up old experiments, and inspect detailed configuration 
settings with pagination support for large configurations.
"""
from itertools   import islice
from pathlib     import Path
from shutil      import rmtree
from thermur.cli import app
from typer       import Argument, Context, Exit, Option, Typer
from typing      import Any, Optional
from yaml        import Loader, load, safe_load

runs = Typer(
    help             = "Explore and manage training runs",
    rich_markup_mode = "rich",
)


@runs.callback(invoke_without_command=True)
def runs_callback(ctx: Context):
    """
    🏃 Explore training runs and configurations.
    
    When called without a subcommand, lists recent training runs.
    Use subcommands to view specific configurations or manage outputs.
    """
    if ctx.invoked_subcommand is None:
        list_runs()


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
        5,
        "--keep", "-k",
        help = "Number of recent runs to keep"
    )
):
    """
    Clean up old training runs.
    
    This command removes old training runs from the outputs directory, keeping
    only the most recent runs based on the specified count. Use dry-run mode
    to preview which directories would be deleted without actually removing them.
    
    Examples:
        thermur runs clean              # Keep 5 most recent
        thermur runs clean -k 10        # Keep 10 most recent  
        thermur runs clean --dry-run    # Preview what would be deleted
    """
    RunsCommand().clean_runs(dry_run, force, keep)


@runs.command("compare")
def compare(
    run1   : str = Argument(..., help="First run ID or path"),
    run2   : str = Argument(..., help="Second run ID or path"),
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
    
    Args:
        run1   : First run identifier (path or "last" for most recent)
        run2   : Second run identifier (path or "last" for most recent)
        domain : Optional domain to filter comparison (e.g., controller)
        
    Examples:
        thermur runs compare last outputs/2025-07-29/15-30-00
        thermur runs compare run1 run2 -d lightning
    """
    RunsCommand().compare_runs(domain, run1, run2)


@runs.command("list")
def list_runs(
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
    RunsCommand().list_runs(limit if not all_runs else None)


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
    
    Examples:
        thermur runs show                    # Show last run
        thermur runs show last               # Explicitly show last  
        thermur runs show outputs/2025-07-30/10-15-30
        thermur runs show -d controller      # Show only controller config
        thermur runs show --all              # Include _system configs
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
    
    def _display_domain_config(
        self,
        config : dict,
        domain : str
    ):
        """
        Display configuration for a single domain with pagination.
        
        Flattens nested configuration dictionaries into dot-notation paths
        and displays them in a paginated table format. Handles both small
        configurations that fit on a single page and large ones requiring
        pagination.
        
        Args:
            config : The configuration dictionary to display
            domain : The configuration domain name (e.g., 'controller')
        """
        if not (items := list(self._flatten_config(config).items())):
            self.ui.print_message(
                message  = f"No configuration found for domain '{domain}'",
                msg_type = "warning"
            )
            return
        
        columns = [
            ("Parameter", "bright_white", 50, "left"),
            ("Value",     "bright_green", 50, "left")
        ]
        page_size   = 20
        total_items = len(items)
        
        if total_items <= page_size:
            self.ui.print_section(
                style = "bright_cyan",
                title = f"{domain.title()} Configuration"
            )
            
            table = self.ui.create_aligned_table(
                border_style = "dim",
                columns      = columns
            )
            
            for path, value in items:
                table.add_row(path, self._format_value(value))
            
            self.ui.display_panel(table)
            
        else:
            self._paginate_config(columns, domain, items, page_size)
    
    def _display_overrides_panel(self, overrides: list[str]):
        """
        Display overrides in a formatted panel.
        
        Creates a warning panel showing all configuration overrides applied
        to a training run, formatted as a bulleted list.
        
        Args:
            overrides: List of override strings from Hydra
        """
        panel  = self.ui.create_warning_panel(
            issues = [f"• {override}" for override in overrides],
            title  = "Configuration Overrides"
        )
        self.ui.display_panel(panel)
    
    def _flatten_config(
        self,
        config : dict,
        prefix : str = ""
    ) -> dict:
        """
        Flatten nested config to dot notation.
        
        Recursively flattens a nested configuration dictionary into a flat
        dictionary with dot-notation keys. Skips private keys (starting with _)
        and stops recursion at instantiated objects (containing _target_).
        
        Args:
            config : Configuration dictionary to flatten
            prefix : Current path prefix for recursion
            
        Returns:
            Flattened dictionary with dot-notation keys
        """
        items = {}
        
        for key, value in config.items():
            if key.startswith('_'):
                continue
            
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict) and '_target_' not in value:
                items.update(self._flatten_config(value, full_key))
            else:
                items[full_key] = value
        
        return items
    
    def _format_overrides(self, overrides_file: Path) -> str:
        """
        Format overrides from file for display.
        
        Reads overrides from a YAML file and formats them for compact display
        in a table. Shows up to two overrides directly and indicates if there
        are more with a count.
        
        Args:
            overrides_file: Path to the overrides.yaml file
            
        Returns:
            Formatted string representation of overrides
        """
        try:
            with open(overrides_file) as f:
                if overrides_list := safe_load(f) or []:
                    formatted = ", ".join(overrides_list[:2])
                    return (
                        f"{formatted} (+{len(overrides_list)-2})"
                        if len(overrides_list) > 2
                        else formatted
                    )
                return "-"
        except Exception:
            return "error reading overrides"
    
    def _format_value(self, value: Any) -> str:
        """
        Format a configuration value for display.
        
        Handles different value types (dict, list, string) and formats them
        appropriately for table display. Long values are truncated with ellipsis.
        
        Args:
            value: Value to format
            
        Returns:
            Formatted string representation suitable for display
        """
        if isinstance(value, dict):
            if '_target_' in value:
                return f"{value['_target_'].split('.')[-1]}(...)"
            else:
                return str(value)
                
        elif isinstance(value, list):
            if len(value) > 3:
                items_preview = ', '.join(str(v) for v in value[:3])
                return f"[{items_preview}, ...] ({len(value)} items)"
            else:
                return str(value)
                
        elif isinstance(value, str) and len(value) > 50:
            return value[:47] + "..."
            
        else:
            return str(value)
    
    def _get_all_runs(self) -> list[Path]:
        """
        Get all run directories sorted by timestamp.
        
        Scans the outputs directory for Hydra run folders (identified by the
        presence of a .hydra subdirectory) and returns them in reverse
        chronological order. This ensures the most recent runs appear first.
        
        Returns:
            List of Path objects pointing to valid run directories
        """
        runs = []
        
        if not self.outputs_dir.exists():
            return runs
            
        for date_dir in sorted(self.outputs_dir.iterdir(), reverse=True):
            if date_dir.is_dir():
                for time_dir in sorted(date_dir.iterdir(), reverse=True):
                    hydra_dir = time_dir / ".hydra"
                    if time_dir.is_dir() and hydra_dir.exists():
                        runs.append(time_dir)
        
        return runs
    
    def _paginate_config(
        self,
        columns   : list[tuple[str, str, int, str]],
        domain    : str,
        items     : list[tuple[str, Any]],
        page_size : int
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
        """
        def render_config_page(
            page_items  : list[tuple[str, Any]], 
            page_num    : int, 
            total_pages : int
        ):
            """
            Render a page of configuration parameters in a table.
            """
            self.ui.print_section(
                style = "bright_cyan",
                title = f"{domain.title()} Configuration "
                        f"(Page {page_num}/{total_pages})"
            )
            
            table = self.ui.create_aligned_table(
                border_style = "dim",
                columns      = columns
            )
            
            for path, value in page_items:
                table.add_row(path, self._format_value(value))
            
            self.ui.display_panel(table)
        
        self.prompts.paginate(
            allow_row_select = False,
            items            = items,
            page_size        = page_size,
            render_page      = render_config_page
        )
    
    def _resolve_run_id(self, run_id: str) -> Path:
        """
        Resolve run ID to actual path.
        
        Handles special "last" alias for the most recent run, or converts
        the provided run ID string to a Path object and validates it exists.
        This provides a consistent way to reference runs by various identifiers.
        
        Args:
            run_id: Either "last" or a path to a run directory
            
        Returns:
            Path object pointing to the resolved run directory
            
        Raises:
            Exit: If no runs exist or specified run is not found
        """
        if run_id == "last":
            runs = self._get_all_runs()
            if not runs:
                self.ui.print_message(
                    message  = "No training runs found in outputs/",
                    msg_type = "error"
                )
                raise Exit(1)
            
            return runs[0]
        
        run_path = Path(run_id)
        if not run_path.exists():
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
        runs = self._get_all_runs()
        
        if len(runs) <= keep:
            self.ui.print_message(
                message  = f"Only {len(runs)} runs found. Nothing to clean.",
                msg_type = "info"
            )
            return
        
        to_delete = runs[keep:]
        
        self.ui.print_header("Runs to Delete")
        for run in to_delete:
            self.ui.print_message(
                message  = f"• {run.relative_to(self.outputs_dir)}",
                msg_type = "info"
            )
        
        if dry_run:
            self.ui.print_message(
                message  = f"Would delete {len(to_delete)} runs (dry run)",
                msg_type = "info"
            )
            return
        
        if not force:
            prompt = f"Delete {len(to_delete)} old runs? This cannot be undone."
            if not self.prompts.confirm(prompt):
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
                try:
                    rmtree(run)
                    deleted += 1
                    progress.update(advance=1, task=task)
                except Exception as e:
                    self.ui.print_message(
                        message  = f"Failed to delete {run}: {e}",
                        msg_type = "error"
                    )
        
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
        run1_path = self._resolve_run_id(run1)
        run2_path = self._resolve_run_id(run2)
        
        config_files = [
            (run1_path / ".hydra" / "config.yaml"),
            (run2_path / ".hydra" / "config.yaml")
        ]
        
        if not all(f.exists() for f in config_files):
            self.ui.print_message(
                message  = "Configuration files not found for one or both runs",
                msg_type = "error"
            )
            raise Exit(1)
        
        cfg1, cfg2 = [
            load(f.open(), Loader) 
            for f in config_files
        ]
        
        cfg1 = {k: v for k, v in cfg1.items() if not k.startswith('_')}
        cfg2 = {k: v for k, v in cfg2.items() if not k.startswith('_')}
        
        self.ui.print_header("Configuration Comparison")
        self.ui.print_message(
            message  = f"Run 1: {run1_path.relative_to(self.outputs_dir)}",
            msg_type = "info"
        )
        self.ui.print_message(
            message  = f"Run 2: {run2_path.relative_to(self.outputs_dir)}",
            msg_type = "info"
        )
        
        all_domains = sorted(set(cfg1.keys()) | set(cfg2.keys()))
        domains     = [domain] if domain else all_domains
        
        for domain in filter(lambda d: d in cfg1 or d in cfg2, domains):
            self.ui.print_section(
                style = "bright_yellow",
                title = f"{domain.title()} Configuration Differences"
            )
            
            flat1 = self._flatten_config(cfg1.get(domain, {}))
            flat2 = self._flatten_config(cfg2.get(domain, {}))
            
            all_keys    = sorted(set(flat1.keys()) | set(flat2.keys()))
            differences = [
                (key, flat1.get(key, "NOT SET"), flat2.get(key, "NOT SET"))
                for key in all_keys
                if flat1.get(key, "NOT SET") != flat2.get(key, "NOT SET")
            ]
            
            if differences:
                table = self.ui.create_aligned_table(
                    border_style = "dim",
                    columns      = [
                        ("Parameter", "bright_white", 40, "left"),
                        ("Run 1",     "bright_green", 30, "left"),
                        ("Run 2",     "bright_blue",  30, "left")
                    ]
                )
                
                for param, val1, val2 in differences:
                    table.add_row(
                        param,
                        self._format_value(val1),
                        self._format_value(val2)
                    )
                
                self.ui.display_panel(table)
            else:
                self.ui.print_message(
                    message  = f"No differences in {dom} configuration",
                    msg_type = "info"
                )
    
    def list_runs(self, limit: Optional[int] = None):
        """
        List recent training runs.
        
        Displays a formatted table of training runs including their IDs,
        timestamps, configuration overrides, and completion status.
        
        Args:
            limit : Maximum number of runs to display (None for all)
        """
        if not (runs := self._get_all_runs()):
            self.ui.print_message(
                message  = "No training runs found. Start training with: "
                           "thermur train",
                msg_type = "info"
            )
            return
        
        display_runs = list(islice(runs, limit)) if limit else runs
        
        columns = [
            ("Date & Time",  "bright_white",  20, "left"),
            ("Overrides",    "bright_green",  30, "left"),
            ("Run ID",       "bright_cyan",   30, "left"),
            ("Status",       "bright_yellow",  8, "center")
        ]
        
        table = self.ui.create_aligned_table(
            border_style = "bright_blue",
            columns      = columns,
            title        = "Training Runs"
        )
        
        for run_path in display_runs:
            run_id = str(run_path.relative_to(self.outputs_dir))
            
            parts = run_id.split('/')
            timestamp = (
                f"{parts[0]} {parts[1].replace('-', ':')}"
                if len(parts) >= 2
                else run_id
            )
            
            overrides_file = run_path / ".hydra" / "overrides.yaml"
            overrides = self._format_overrides(overrides_file) if overrides_file.exists() else "-"
            status = "✓" if (run_path / "training_complete").exists() else "..."
            
            table.add_row(timestamp, overrides, run_id, status)
        
        self.ui.display_panel(table)
        
        if limit and len(runs) > limit:
            self.ui.print_message(
                message  = f"Showing {limit} of {len(runs)} runs. "
                           f"Use --all to see all runs.",
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
        run_path    = self._resolve_run_id(run_id)
        config_file = run_path / ".hydra" / "config.yaml"
        
        if not config_file.exists():
            self.ui.print_message(
                message  = f"No configuration found for run: {run_id}",
                msg_type = "error"
            )
            raise Exit(1)
        
        with open(config_file) as f:
            cfg = load(f, Loader=Loader)
        
        cfg = (
            cfg if include_system
            else {k: v for k, v in cfg.items() if not k.startswith('_')}
        )
        
        self.ui.print_header(
            f"Configuration: {run_path.relative_to(self.outputs_dir)}"
        )
        
        if (overrides_file := run_path / ".hydra" / "overrides.yaml").exists():
            with open(overrides_file) as f:
                if overrides := safe_load(f) or []:
                    self._display_overrides_panel(overrides)
        
        domains = [domain] if domain else sorted(cfg.keys())
        
        for i, domain in enumerate(domains):
            if domain not in cfg:
                self.ui.print_message(
                    message  = f"Domain '{domain}' not found in configuration",
                    msg_type = "warning"
                )
                continue
            
            self._display_domain_config(cfg[domain], domain)
            
            if (
                len(domains)     > 1 and 
                i < len(domains) - 1 and 
                not self.prompts.confirm("Show next domain?")
            ):
                break