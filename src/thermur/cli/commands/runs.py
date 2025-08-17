"""
Manage and explore training runs and their configurations.

This module provides the 'runs' command for interacting with Hydra's output
directories. It enables users to list training runs, compare configurations
between runs, clean up old experiments, and inspect detailed configuration
settings with pagination support for large configurations.
"""
from config.types import ConfigItem, TableColumn
from contextlib   import contextmanager, suppress
from datetime     import datetime
from pathlib      import Path
from yaml         import safe_load
from shutil       import rmtree
from thermur.cli  import app
from typer        import Argument, Context, Exit, Option, Typer
from typing       import Any, Iterator

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
            yield safe_load(f)

    except Exception as e:
        app.ui.print_message(f"Failed to load {path}: {e}", "error")
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
    cmd = RunsCommand()
    cmd.dry_run = dry_run
    cmd.force   = force
    cmd.clean_runs(keep)


@runs.command("compare")
def compare(
    run1   : str | None = Argument(
        None,
        help="First run ID or path. Use 'last[N]' for Nth most recent (defaults to 'last[1]')"
    ),
    run2   : str | None = Argument(
        None,
        help="Second run ID or path. Use 'last[N]' for Nth most recent (defaults to 'last[2]')"
    ),
    domain : str | None = Option(
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
    cmd        = RunsCommand()
    cmd.domain = domain
    cmd.compare_runs(run1, run2)


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
    RunsCommand().list_runs(None if all_runs else limit, True)


@runs.command("show")
def show(
    run_id: str | None = Argument(
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
    domain: str | None = Option(
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
    cmd = RunsCommand()
    cmd.domain         = domain
    cmd.include_system = all_configs
    cmd.run_id         = run_id or "last"
    cmd.show_config()


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
        self.cfg            = app.cfg
        self.outputs_dir    = Path("outputs")
        self.prompts        = app.prompts
        self.system         = app.system
        self.ui             = app.ui

        self.domain         : str | None       = None
        self.run_id         : str              = "last"
        self.dry_run        : bool             = False
        self.force          : bool             = False
        self.include_system : bool             = False
        self.overrides      : list[str] | None = None
        self.run_path       : Path | None      = None

    def _display_config(
        self,
        items      : list[ConfigItem],
        title      : str,
        page_size  : int  = 20,
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
        self.ui.print_header("Training Run Management")

        if self.run_path:
            self.ui.print_section(
                f"Configuration: {self.run_path.relative_to(self.outputs_dir)}",
                True
            )
            
            if wandb_url := self._get_wandb_url(self.run_path):
                self.ui.print_message(
                    message  = f" WandB dashboard: {wandb_url}",
                    msg_type = "magic"
                )
                self.ui.console.print()

        if using_last:
            self.ui.print_message(
                message  = (
                    "Using most recent run. "
                    "(Use specific run ID to view others)"
                ),
                msg_type = "info"
            )

        if self.overrides:
            self._display_overrides_panel()

        def render_page(
            page_items  : list[ConfigItem],
            page_num    : int,
            total_pages : int
        ):
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
                    TableColumn("center", "bright_yellow", "",          2),
                    TableColumn("left",   "bright_white",  "Parameter", 48),
                    TableColumn("left",   "bright_green",  "Value",     50)
                ]
            )

            for i in page_items:
                table.add_row(
                    "●" if i.is_override else " ",
                    i.path,
                    self.ui.format_truncated(value=i.value)
                )
            self.ui.display_panel(table)

        self.prompts.paginate(
            allow_row_select = False,
            items            = sorted(items, key=lambda x: x.path),
            page_size        = page_size,
            render_page      = render_page
        )

    def _display_overrides_panel(self):
        """
        Display overrides in a formatted panel.

        Creates a warning panel showing all configuration overrides applied
        to a training run, formatted as a bulleted list.
        """
        if self.overrides is None:
            return

        panel = self.ui.create_warning_panel(
            issues = self.overrides,
            title  = "Configuration Overrides"
        )
        self.ui.display_panel(panel)

    def _display_runs_table(
        self,
        columns      : list[TableColumn],
        runs         : list[tuple[Path, float]],
        border_style : str = "bright_blue"
    ):
        """
        Display runs in a formatted table with run ID, overrides, and status.

        Args:
            columns      : Column definitions for the table
            runs         : List of (path, timestamp) tuples
            border_style : Border style for the table
        """
        table = self.ui.create_aligned_table(
            border_style = border_style,
            columns      = columns
        )

        [
            table.add_row(
                str(run_path.relative_to(self.outputs_dir)),
                datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M"),
                self.ui.format_summary_list(overrides) 
                if (overrides := self._load_overrides(run_path)) else "-",
                self.ui.format_run_status(run_path)
            )
            for run_path, timestamp in runs
        ]

        self.ui.display_panel(table)

    def _flatten_config(
        self,
        config : dict[str, Any],
        prefix : str = ""
    ) -> dict[str, ConfigItem]:
        """
        Flatten nested config to dot notation.

        Recursively flattens a nested configuration dictionary into a flat
        dictionary with dot-notation keys. Expands model instances to show
        their parameters. Tracks which values were overridden.

        Args:
            config : Configuration dictionary to flatten
            prefix : Current path prefix for recursion

        Returns:
            Flattened dictionary with dot-notation keys and override info
        """
        items        : dict[str, ConfigItem] = {}
        override_set : set[str] = {
            override.split('=')[0].strip('+')
            for override in self.overrides
            if '=' in override
        } if self.overrides else set()

        for key, value in config.items():
            if key.startswith('_') and key != '_target_':
                continue

            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                if '_target_' in value:
                    value_dict   : dict[str, Any] = value
                    model_params : dict[str, Any] = {
                        k: v for k, v in value_dict.items() if k != '_target_'
                    }
                    if not model_params:
                        target_value = str(value_dict['_target_'])
                        items[full_key] = ConfigItem(
                            is_override = False,
                            path        = full_key,
                            value       = target_value.split('.')[-1]
                        )
                        continue
                    value = model_params

                items.update(self._flatten_config(value, full_key))
            else:
                is_override = full_key in override_set or any(
                    full_key.startswith(p) for p in override_set
                )
                items[full_key] = ConfigItem(is_override, full_key, value)

        return items

    def _get_all_runs(self) -> list[tuple[Path, float]]:
        """
        Get all run directories with their timestamps, sorted by timestamp.

        Scans the outputs directory for Hydra run folders (identified by the
        presence of a .hydra subdirectory) and returns them in reverse
        chronological order. This ensures the most recent runs appear first.

        Returns:
            List of (Path, timestamp) tuples sorted by timestamp
        """
        if not self.outputs_dir.exists():
            return []

        return sorted(
            [
                (run_path, timestamp)
                for hydra_dir in self.outputs_dir.glob("**/.hydra")
                if (run_path  := hydra_dir.parent)
                for config_file in [run_path / ".hydra" / "config.yaml"]
                if (timestamp := (
                    config_file.stat().st_ctime if config_file.exists()
                    else run_path.stat().st_ctime
                ))
            ],
            key     = lambda x: x[1],
            reverse = True
        )

    def _get_domains(self, cfg: dict[str, Any]) -> list[str]:
        """
        Get list of domains to process from configuration.

        Args:
            cfg: Configuration dictionary

        Returns:
            List of domain names to process
        """
        if self.domain:
            if self.domain not in cfg:
                self.ui.print_message(
                    message  = f"Domain '{self.domain}' not found in configuration",
                    msg_type = "warning"
                )
                return []
            return [self.domain]
        return sorted(cfg.keys())
    
    def _get_wandb_url(self, run_path: Path) -> str | None:
        """
        Get WandB dashboard URL for a run if available.

        Args:
            run_path: Path to the run directory

        Returns:
            WandB URL string or None if not available
        """
        if not (run_file := run_path / "wandb" / "latest-run" / "run.txt").exists():
            return None

        try:
            wandb_run_id = run_file.read_text().strip()
            
            cfg       = self._load_run_config(run_path)
            wandb_cfg = cfg.get("wandb", {})
            project   = wandb_cfg.get("project", "thermur-imitation")
            base_url  = f"https://wandb.ai"
            return (
                f"{base_url}/{entity}/{project}/runs/{wandb_run_id}"
                if (entity := wandb_cfg.get("entity"))
                else f"{base_url}/{project}/runs/{wandb_run_id}"
            )
                
        except Exception:
            return None

    def _load_overrides(self, run_path: Path) -> list[str]:
        """
        Load override configuration from a run.

        Args:
            run_path : Path to run directory

        Returns:
            List of override strings, or empty list if none found
        """
        overrides_file = run_path / ".hydra" / "overrides.yaml"
        if overrides_file.exists():
            with suppress(Exception), open(overrides_file) as f:
                return safe_load(f) or []
        return []

    def _load_run_config(self, run_path: Path) -> dict[str, Any]:
        """
        Load and optionally filter a run's configuration.

        Args:
            run_path: Path to run directory

        Returns:
            Configuration dictionary
        """
        with load_yaml(run_path / ".hydra" / "config.yaml") as cfg:
            return cfg if self.include_system else {
                k: v for k, v in cfg.items() if not k.startswith('_')
            }

    def _resolve_run_id(self, run_id: str) -> Path:
        """
        Resolve run ID to actual path.

        Handles special "last" alias for the most recent run, "last[N]" for the
        Nth most recent run, or converts the provided run ID string to a Path
        object and validates it exists. This provides a consistent way to
        reference runs by various identifiers.

        Args:
            run_id: Either "last", "last[N]", or a path to a run directory

        Returns:
            Path object pointing to the resolved run directory

        Raises:
            Exit: If no runs exist or specified run is not found
        """
        if not run_id.startswith("last"):
            if not (run_path := Path(run_id)).exists():
                self.ui.print_message(f"Run not found: {run_id}", "error")
                raise Exit(1)
            return run_path

        if not (runs := self._get_all_runs()):
            self.ui.print_message("No training runs found in outputs/", "error")
            raise Exit(1)

        match run_id:
            case "last":
                n = 1
            case _ if run_id.startswith("last[") and run_id.endswith("]"):
                try:
                    if (n := int(run_id[5:-1])) < 1:
                        raise ValueError
                except ValueError:
                    self.ui.print_message(
                        message  = f"Invalid syntax: {run_id}. N must be positive.",
                        msg_type = "error"
                    )
                    raise Exit(1)
            case _:
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
                message  = f"Only {len(runs)} runs found. Cannot get run #{n}",
                msg_type = "error"
            )
            raise Exit(1)

        return runs[n - 1][0]

    def clean_runs(self, keep: int):
        """
        Clean up old training runs.

        Removes old training runs from the outputs directory, keeping only
        the specified number of most recent runs. Provides safety features
        including dry-run mode and confirmation prompts.

        Args:
            keep: Number of recent runs to keep
        """
        self.ui.print_header("Training Run Management")

        if len(runs := self._get_all_runs()) <= keep:
            self.ui.print_message(
                message  = f"Only {len(runs)} runs found. Nothing to clean.",
                msg_type = "info"
            )
            return

        to_delete = runs[keep:]

        self.ui.print_section("Runs to Delete", minor=True)

        columns = [
            TableColumn("left",   "bright_red",   "Run ID",  35),
            TableColumn("left",   "bright_white", "Started", 20),
            TableColumn("center", "bright_white", "Status",  10)
        ]

        self.ui.print_message(
            message  = (
                f"{len(to_delete)} run{'s' if len(to_delete) > 1 else ''} "
                f"will be deleted"
            ),
            msg_type = "warning"
        )

        self._display_runs_table(
            columns      = columns,
            runs         = to_delete
        )

        if self.dry_run:
            self.ui.print_message(
                message  = f"Would delete {len(to_delete)} runs (dry run mode)",
                msg_type = "info"
            )
            return

        if not self.prompts.confirm_deletion(
            count = len(to_delete),
            items = "training runs",
            keep  = keep,
            force = self.force
        ):
            self.ui.print_message(
                message  = "Cleanup cancelled",
                msg_type = "warning"
            )
            return

        deleted = 0
        with (progress := self.ui.create_thermal_progress()):
            task = progress.add_task("Deleting runs...", total=len(to_delete))
            for run_path, _ in to_delete:
                with suppress(Exception):
                    rmtree(run_path)
                    deleted += 1
                    progress.update(task, advance=1)

        self.ui.print_message(
            message  = f"Deleted {deleted} old runs",
            msg_type = "success"
        )

    def compare_runs(
        self,
        run1 : str | None = None,
        run2 : str | None = None
    ):
        """
        Compare configurations between two runs.

        Displays side-by-side differences between the configurations of two
        training runs. Can be filtered to show only a specific domain.

        The default compares the last two runs. If only `run1` is provided, the
        last run is compared against it.

        Args:
            run1 : First run identifier (defaults to last[1])
            run2 : Second run identifier (defaults to last[2])
        """
        if run1 and not run2:
            run2 = "last[1]"
        else:
            run1 = run1 or "last[1]"
            run2 = run2 or "last[2]"

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

        all_configs = cfg1 | cfg2
        domains: list[tuple[str, list[tuple[str, Any, Any]]]] = []

        for domain in (d for d in self._get_domains(all_configs) if d in all_configs):
            flat1 = self._flatten_config(cfg1.get(domain, {}))
            flat2 = self._flatten_config(cfg2.get(domain, {}))

            differences: list[tuple[str, Any, Any]] = [
                (key, val1, val2)
                for key in sorted(set(flat1) | set(flat2))
                if (val1 := flat1.get(key, ConfigItem(False, "", "NOT SET")).value) !=
                   (val2 := flat2.get(key, ConfigItem(False, "", "NOT SET")).value)
            ]
            if differences:
                domains.append((domain, differences))

        if not domains:
            self.ui.print_message(
                message  = "No configuration differences found between runs",
                msg_type = "info"
            )
            return

        for domain, differences in domains:
            table = self.ui.create_aligned_table(
                border_style = "dim",
                columns      = [
                    TableColumn("left", "bright_white", "Parameter", 40),
                    TableColumn("left", "bright_green", "Run 1",     30),
                    TableColumn("left", "bright_blue",  "Run 2",     30)
                ],
                title = f"{domain.title()} Configuration"
            )

            for param, val1, val2 in differences:
                table.add_row(
                    param,
                    self.ui.format_truncated(value=val1),
                    self.ui.format_truncated(value=val2)
                )

            self.ui.display_panel(table)

    def list_runs(
        self,
        limit       : int | None = None,
        show_header : bool          = True
    ):
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

        if not (runs := self._get_all_runs()):
            self.ui.print_message(
                message  = "No training runs found. Start training with: "
                           "thermur train",
                msg_type = "info"
            )
            return

        display_runs = runs[:limit] if limit else runs

        if limit:
            message = (
                f"Showing {limit} of {len(runs)} runs. (Use --all to see all runs)"
                if len(runs) > limit else f"Showing all {len(runs)} runs."
            )
            self.ui.print_message(message, "info")

        self.ui.print_section("Recent Runs", minor=True)
        self.ui.display_status_legend()

        columns = [
            TableColumn("left",   "bright_cyan",  "Run ID",    25),
            TableColumn("left",   "bright_white", "Started",   20),
            TableColumn("left",   "bright_green", "Overrides", 45),
            TableColumn("center", "bright_white", "Status",    10)
        ]

        self._display_runs_table(
            columns = columns,
            runs    = display_runs
        )

    def show_config(self):
        """
        Display configuration for a run with pagination.

        Shows the full Hydra configuration for a training run, optionally
        filtered by domain. Large configurations are automatically paginated
        for better readability.
        """
        self.run_path  = self._resolve_run_id(self.run_id)
        cfg            = self._load_run_config(self.run_path)
        self.overrides = self._load_overrides(self.run_path)

        if not (domains := self._get_domains(cfg)):
            return

        if not (all_items := [
            item
            for domain in domains
            for item in self._flatten_config(cfg[domain], domain).values()
        ]):
            self.ui.print_message(
                message  = "No configuration parameters found",
                msg_type = "info"
            )
            return

        self._display_config(
            items      = all_items,
            title      = "Configuration Parameters",
            using_last = self.run_id == "last"
        )
