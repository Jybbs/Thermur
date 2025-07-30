"""
Manage and explore training runs and their configurations.

This command provides easy access to Hydra's output directories, allowing users
to view configurations, compare runs, and clean up old experiments.
"""
from pathlib     import Path
from shutil      import rmtree
from thermur.cli import app
from typer       import Argument, Context, Exit, Option, Typer
from typing      import Optional
from yaml        import safe_load, load, Loader

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
    
    Examples:
        thermur runs clean              # Keep 5 most recent
        thermur runs clean -k 10        # Keep 10 most recent
        thermur runs clean --dry-run    # Preview what would be deleted
    """
    command = RunsCommand()
    command.clean_runs(keep=keep, dry_run=dry_run, force=force)


@runs.command("compare")
def compare(
    run1: str = Argument(..., help="First run ID or path"),
    run2: str = Argument(..., help="Second run ID or path"),
    domain: Optional[str] = Option(
        None,
        "--domain", "-d",
        help = "Compare only specific domain"
    )
):
    """
    Compare configurations between two training runs.
    
    Examples:
        thermur runs compare last outputs/2025-07-29/15-30-00
        thermur runs compare run1 run2 -d lightning
    """
    command = RunsCommand()
    command.compare_runs(run1, run2, domain)


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
    
    Examples:
        thermur runs list           # Show 10 most recent
        thermur runs list -n 20     # Show 20 most recent
        thermur runs list --all     # Show all runs
    """
    command = RunsCommand()
    command.list_runs(limit=limit if not all_runs else None)


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
    
    Examples:
        thermur runs show                    # Show last run
        thermur runs show last               # Explicitly show last  
        thermur runs show outputs/2025-07-30/10-15-30
        thermur runs show -d controller      # Show only controller config
        thermur runs show --all              # Include _system configs
    """
    command = RunsCommand()
    command.show_config(
        run_id         = run_id or "last",
        domain         = domain,
        include_system = all_configs
    )


class RunsCommand:
    """Encapsulates the state and logic for the 'runs' command."""
    
    def __init__(self):
        """Initializes the command with shared context components."""
        self.cfg         = app.cfg
        self.outputs_dir = Path("outputs")
        self.prompts     = app.prompts
        self.system      = app.system
        self.ui          = app.ui
    
    def _display_domain_config(self, domain: str, config: dict):
        """
        Display configuration for a single domain with pagination.
        
        Flattens nested configuration dictionaries into dot-notation paths
        and displays them in a paginated table format.
        
        Args:
            domain: The configuration domain name
            config: The configuration dictionary to display
        """
        flat_config = self._flatten_config(config)
        items       = list(flat_config.items())
        
        if not items:
            self.ui.print_message(
                message  = f"No configuration found for domain '{domain}'",
                msg_type = "warning"
            )
            return
        
        page_size   = 20
        total_items = len(items)
        
        if total_items <= page_size:
            self.ui.print_section(
                title = f"{domain.title()} Configuration",
                style = "bright_cyan"
            )
            
            columns = [
                ("Parameter", "bright_white", 50, "left"),
                ("Value", "bright_green", 50, "left")
            ]
            
            table = self.ui.create_aligned_table(
                columns      = columns,
                border_style = "dim"
            )
            
            for path, value in items:
                table.add_row(path, self._format_value(value))
            
            self.ui.display_panel(table)
        else:
            current_page = 1
            total_pages  = (total_items + page_size - 1) // page_size
            
            while True:
                start = (current_page - 1) * page_size
                end   = min(start + page_size, total_items)
                
                import os
                os.system('clear' if os.name == 'posix' else 'cls')
                
                self.ui.print_section(
                    title = f"{domain.title()} Configuration "
                            f"(Page {current_page}/{total_pages})",
                    style = "bright_cyan"
                )
                
                columns = [
                    ("Parameter", "bright_white", 50, "left"),
                    ("Value", "bright_green", 50, "left")
                ]
                
                table = self.ui.create_aligned_table(
                    columns      = columns,
                    border_style = "dim"
                )
                
                for path, value in items[start:end]:
                    table.add_row(path, self._format_value(value))
                
                self.ui.display_panel(table)
                
                nav_options = []
                if current_page > 1:
                    nav_options.append(("[P]revious", "p"))
                if current_page < total_pages:
                    nav_options.append(("[N]ext", "n"))
                nav_options.append(("[Q]uit", "q"))
                
                nav_text = " | ".join([opt[0] for opt in nav_options])
                self.ui.print_message(
                    message  = nav_text,
                    msg_type = "info"
                )
                
                choice = self.prompts.get_choice_input(
                    [opt[1] for opt in nav_options]
                )
                
                if choice == "p" and current_page > 1:
                    current_page -= 1
                elif choice == "n" and current_page < total_pages:
                    current_page += 1
                elif choice == "q":
                    break
    
    def _display_overrides_panel(self, overrides: list[str]):
        """Display overrides in a formatted panel."""
        issues = [f"• {override}" for override in overrides]
        panel  = self.ui.create_warning_panel(
            title  = "Configuration Overrides",
            issues = issues
        )
        self.ui.display_panel(panel)
    
    def _flatten_config(self, config: dict, prefix: str = "") -> dict:
        """
        Flatten nested config to dot notation.
        
        Args:
            config: Configuration dictionary to flatten
            prefix: Current path prefix for recursion
            
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
    
    def _format_value(self, value) -> str:
        """
        Format a configuration value for display.
        
        Args:
            value: Value to format
            
        Returns:
            Formatted string representation
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
        chronological order.
        
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
    
    def _resolve_run_id(self, run_id: str) -> Path:
        """
        Resolve run ID to actual path.
        
        Handles special "last" alias for the most recent run, or converts
        the provided run ID string to a Path object and validates it exists.
        
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
    
    def clean_runs(self, keep: int, dry_run: bool, force: bool):
        """
        Clean up old training runs.
        
        Args:
            keep: Number of recent runs to keep
            dry_run: Whether to simulate deletion without actually deleting
            force: Whether to skip confirmation prompt
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
                    progress.update(task, advance=1)
                except Exception as e:
                    self.ui.print_message(
                        message  = f"Failed to delete {run}: {e}",
                        msg_type = "error"
                    )
        
        self.ui.print_message(
            message  = f"Deleted {deleted} old runs",
            msg_type = "success"
        )
    
    def compare_runs(self, run1: str, run2: str, domain: Optional[str] = None):
        """
        Compare configurations between two runs.
        
        Args:
            run1: First run identifier
            run2: Second run identifier  
            domain: Optional domain to filter comparison
        """
        run1_path = self._resolve_run_id(run1)
        run2_path = self._resolve_run_id(run2)
        
        config1_file = run1_path / ".hydra" / "config.yaml"
        config2_file = run2_path / ".hydra" / "config.yaml"
        
        if not config1_file.exists() or not config2_file.exists():
            self.ui.print_message(
                message  = "Configuration files not found for one or both runs",
                msg_type = "error"
            )
            raise Exit(1)
        
        with open(config1_file) as f:
            cfg1 = load(f, Loader=Loader)
        with open(config2_file) as f:
            cfg2 = load(f, Loader=Loader)
        
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
        
        for dom in domains:
            if dom not in cfg1 and dom not in cfg2:
                continue
                
            self.ui.print_section(
                title = f"{dom.title()} Configuration Differences",
                style = "bright_yellow"
            )
            
            flat1 = self._flatten_config(cfg1.get(dom, {}))
            flat2 = self._flatten_config(cfg2.get(dom, {}))
            
            all_keys    = sorted(set(flat1.keys()) | set(flat2.keys()))
            differences = []
            
            for key in all_keys:
                val1 = flat1.get(key, "NOT SET")
                val2 = flat2.get(key, "NOT SET")
                
                if val1 != val2:
                    differences.append((key, val1, val2))
            
            if differences:
                columns = [
                    ("Parameter", "bright_white", 40, "left"),
                    ("Run 1", "bright_green", 30, "left"),
                    ("Run 2", "bright_blue", 30, "left")
                ]
                
                table = self.ui.create_aligned_table(
                    columns      = columns,
                    border_style = "dim"
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
            limit: Maximum number of runs to display (None for all)
        """
        runs = self._get_all_runs()
        
        if not runs:
            self.ui.print_message(
                message  = "No training runs found. Start training with: "
                           "thermur train",
                msg_type = "info"
            )
            return
        
        display_runs = runs[:limit] if limit else runs
        
        columns = [
            ("Run ID", "bright_cyan", 30, "left"),
            ("Date & Time", "bright_white", 20, "left"),
            ("Overrides", "bright_green", 30, "left"),
            ("Status", "bright_yellow", 8, "center")
        ]
        
        table = self.ui.create_aligned_table(
            title        = "Training Runs",
            columns      = columns,
            border_style = "bright_blue"
        )
        
        for run_path in display_runs:
            run_id = str(run_path.relative_to(self.outputs_dir))
            
            parts = run_id.split('/')
            if len(parts) >= 2:
                date_part = parts[0]
                time_part = parts[1]
                timestamp = f"{date_part} {time_part.replace('-', ':')}"
            else:
                timestamp = run_id
            
            overrides_file = run_path / ".hydra" / "overrides.yaml"
            overrides      = ""
            
            if overrides_file.exists():
                try:
                    with open(overrides_file) as f:
                        overrides_list = safe_load(f) or []
                        if overrides_list:
                            overrides = ", ".join(overrides_list[:2])
                            if len(overrides_list) > 2:
                                overrides += f" (+{len(overrides_list)-2})"
                except Exception:
                    overrides = "error reading overrides"
            
            status = "✓" if (run_path / "training_complete").exists() else "..."
            
            table.add_row(run_id, timestamp, overrides or "-", status)
        
        self.ui.display_panel(table)
        
        if limit and len(runs) > limit:
            self.ui.print_message(
                message  = f"Showing {limit} of {len(runs)} runs. "
                           f"Use --all to see all runs.",
                msg_type = "info"
            )
    
    def show_config(
        self,
        run_id: str,
        domain: Optional[str] = None,
        include_system: bool = False
    ):
        """
        Display configuration for a run with pagination.
        
        Shows the full Hydra configuration for a training run, optionally
        filtered by domain. Large configurations are automatically paginated
        for better readability.
        
        Args:
            run_id: Run identifier or "last" for most recent
            domain: Optional specific domain to display
            include_system: Whether to include system (_) configurations
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
        
        if not include_system:
            cfg = {k: v for k, v in cfg.items() if not k.startswith('_')}
        
        relative_path = run_path.relative_to(self.outputs_dir)
        self.ui.print_header(f"Configuration: {relative_path}")
        
        overrides_file = run_path / ".hydra" / "overrides.yaml"
        if overrides_file.exists():
            with open(overrides_file) as f:
                overrides = safe_load(f) or []
                if overrides:
                    self._display_overrides_panel(overrides)
        
        domains = [domain] if domain else sorted(cfg.keys())
        
        for i, dom in enumerate(domains):
            if dom not in cfg:
                self.ui.print_message(
                    message  = f"Domain '{dom}' not found in configuration",
                    msg_type = "warning"
                )
                continue
            
            self._display_domain_config(dom, cfg[dom])
            
            if len(domains) > 1 and i < len(domains) - 1:
                if not self.prompts.confirm("Show next domain?"):
                    break