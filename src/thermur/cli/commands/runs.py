"""
Manage and explore training runs and their configurations.

This module provides the 'runs' command for interacting with training runs
via the WandB API. It enables users to list training runs, compare configurations
between runs, and inspect detailed configuration settings with pagination support
for large configurations.
"""
from __future__   import annotations
from config.types import CfgItem, TableColumn
from datetime     import datetime
from json         import loads
from sys          import stdin
from thermur.cli  import app
from typer        import Argument, Context, Exit, Option, Typer
from typing       import TYPE_CHECKING

if TYPE_CHECKING:
    from typing                 import Any
    from wandb.apis.public.runs import Run

import wandb

runs = Typer(
    help             = "🏃 Explore training runs and configurations",
    rich_markup_mode = "rich",
)


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


@runs.command("compare")
def compare(
    run1: str | None = Argument(
        None,
        help = (
            "First run ID or name. Use 'last[N]' for Nth most recent "
            "(defaults to 'last')"
        )
    ),
    run2: str | None = Argument(
        None,
        help = (
            "Second run ID or name. Use 'last[N]' for Nth most recent "
            "(defaults to 'last[2]')"
        )
    ),
    build: str | None = Option(
        None,
        "--build", "-b",
        help = "Compare only specific build (e.g., controller, training)"
    )
):
    """
    Compare configurations between two runs. Defaults to last 2.

    Displays side-by-side differences between the configurations of two
    training runs. You can filter the comparison to a specific build
    (e.g., controller, training) to focus on relevant settings.

    When no arguments are provided, compares the two most recent runs.
    You can use 'last[N]' syntax to reference the Nth most recent run.

    Examples:
        thermur runs compare                    # Compare last 2 runs
        thermur runs compare last TH0001        # Compare most recent to specific
        thermur runs compare last[1] last[3]    # 1st vs 3rd most recent
        thermur runs compare TH0001 TH0002      # Compare specific runs
        thermur runs compare last -b controller # Compare specific build
    """
    cmd       = RunsCommand()
    cmd.build = build
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
            "Run ID or name. Use 'last' for most recent, 'last[N]' for Nth "
            "most recent (defaults to 'last')"
        )
    ),
    all_cfgs: bool = Option(
        False,
        "--all", "-a",
        help = "Include system (_) configurations"
    ),
    build: str | None = Option(
        None,
        "--build", "-b",
        help = "Show only specific build (e.g., controller, training)"
    ),
    dry_run: bool = Option(
        False,
        "--dry-run", "-d",
        help = "Show config from stdin (for dry-run mode)"
    )
):
    """
    Display run configuration. Use 'last[N]' for Nth most recent.

    Shows the full configuration for a training run, optionally filtered
    by build. Large configurations are automatically paginated for better
    readability. System configurations (prefixed with _) are hidden by default
    but can be included with the --all flag.

    Examples:
        thermur runs show               # Show last run
        thermur runs show last          # Explicitly show last
        thermur runs show last[2]       # Show 2nd most recent run
        thermur runs show TH0001        # Show a specific run
        thermur runs show -b controller # Show only controller config
        thermur runs show --all         # Include system (_) configs
        thermur runs show --dry-run     # Show config from stdin (dry-run mode)
    """
    cmd                = RunsCommand()
    cmd.build          = build
    cmd.include_system = all_cfgs
    cmd.run_id         = run_id or "last"
    
    cmd.show_cfg(dry_run)


class RunsCommand:
    """
    Encapsulates the state and logic for the 'runs' command.

    This class provides methods for listing, comparing, and displaying
    training run configurations from the WandB API. Private methods
    handle internal operations like API access and formatting.
    """

    def __init__(self):
        """
        Initialize the command with shared context components.
        
        Sets up WandB API for accessing training run data.
        """
        self.api     = wandb.Api()
        self.cfg     = app.cfg
        self.project = self.cfg.wandb.project
        self.prompts = app.prompts
        self.ui      = app.ui

        self.build          : str | None       = None
        self.include_system : bool             = False
        self.overrides      : list[str] | None = None
        self.run_id         : str              = "last"
        self.run_name       : str | None       = None

    def _api_runs(
        self, 
        per_page : int            | None = None,
        filters  : dict[str, Any] | None = None
    ) -> list[Run]:
        """
        Wrapper for consistent API runs calls.
        
        Centralizes the common pattern of calling api.runs with project
        and ordering parameters.
        
        Args:
            per_page : Number of runs to fetch
            filters  : Optional filters for the query
            
        Returns:
            List of Run objects
        """
        return list(
            self.api.runs(
                filters  = filters,
                order    = "-created_at",
                path     = self.project,
                per_page = per_page or 1000
            )
        )

    def _display_cfg(
        self,
        items      : list[CfgItem],
        title      : str,
        page_size  : int  = 20,
        using_last : bool = False
    ):
        """
        Unified config display using pagination for all cases.

        Args:
            items      : List of CfgItem tuples
            title      : Title for the configuration section
            page_size  : Number of items per page (default 20)
            using_last : Whether the user is viewing the last run
        """
        self.ui.print_header("Training Run Management")

        if self.run_name:
            self.ui.print_section(
                f"Configuration: {self.run_name}",
                True
            )
            
            if wandb_url := self._get_wandb_url(self.run_name):
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
            page_items  : list[CfgItem],
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
        columns : list[TableColumn],
        runs    : list[Run]
    ):
        """
        Display runs in a formatted table with run ID, overrides, and status.

        Formats WandB run objects into a Rich table showing run names/IDs,
        creation timestamps, configuration overrides, and completion status icons.

        Args:
            columns : Column definitions for the table
            runs    : List of WandB run objects
        """
        table = self.ui.create_aligned_table(
            border_style = "bright_blue",
            columns      = columns
        )

        for run in runs:
            overrides = run.config.get("overrides", [])
            timestamp = datetime.fromisoformat(
                run
                .created_at.replace("T", " ")
                .replace("Z", "+00:00")
            ).strftime("%Y-%m-%d %H:%M")

            status_icon = {
                "finished" : "✓",
                "failed"   : "✗",
                "crashed"  : "✗",
                "running"  : "⟳",
                "killed"   : "✗"
            }.get(run.state, "?")
            
            table.add_row(
                run.name or run.id[:8],
                timestamp,
                self.ui.format_summary_list(overrides) if overrides else "-",
                status_icon
            )

        self.ui.display_panel(table)

    def _ensure_run_exists(self, run_id: str) -> Run:
        """
        Get a run by name/ID or exit with error.
        
        Provides consistent error handling for run lookups across commands.
        
        Args:
            run_id: Run identifier to look up
            
        Returns:
            WandB run object
            
        Raises:
            Exit: If run is not found
        """
        if not (run := self._get_run_by_name(run_id)):
            self.ui.print_message(f"Run not found: {run_id}", "error")
            raise Exit(1)
        
        return run
    
    def _extract_training_sections(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """
        Extract the three main training configuration sections.
        
        Filters configuration to only include controller, environment, 
        and training sections that are present in the source config.
        
        Args:
            cfg: Source configuration dictionary
            
        Returns:
            Dictionary with only the training-related sections
        """
        return {
            k: cfg[k] 
            for k in ["controller", "environment", "training"]
            if  k in cfg
        }

    def _flatten_cfg(
        self,
        cfg    : dict[str, Any],
        prefix : str = ""
    ) -> dict[str, CfgItem]:
        """
        Flatten nested config to dot notation.

        Recursively flattens a nested configuration dictionary into a flat
        dictionary with dot-notation keys. Expands model instances to show
        their parameters. Tracks which values were overridden.

        Args:
            cfg    : Configuration dictionary to flatten
            prefix : Current path prefix for recursion

        Returns:
            Flattened dictionary with dot-notation keys and override info
        """
        items        : dict[str, CfgItem] = {}
        override_set : set[str] = {
            override.split('=')[0].strip('+')
            for override in self.overrides
            if '=' in override
        } if self.overrides else set()

        for key, value in cfg.items():
            if key.startswith('_') and key != '_target_':
                continue

            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                if '_target_' in value:
                    if not (model_params := {
                        k: v for k, v in value.items() if k != '_target_'
                    }):
                        items[full_key] = CfgItem(
                            is_override = False,
                            path        = full_key,
                            value       = str(value['_target_']).split('.')[-1]
                        )
                        continue
                    value = model_params

                items.update(self._flatten_cfg(value, full_key))
            else:
                is_override = full_key in override_set or any(
                    full_key.startswith(p) for p in override_set
                )
                items[full_key] = CfgItem(is_override, full_key, value)

        return items

    def _get_builds(self, cfg: dict[str, Any]) -> list[str]:
        """
        Get list of builds to process from configuration.

        Args:
            cfg: Configuration dictionary

        Returns:
            List of build names to process
        """
        if not self.build:
            return sorted(cfg.keys())
        
        if self.build not in cfg:
            self.ui.print_message(
                message  = f"Build '{self.build}' not found in configuration",
                msg_type = "warning"
            )
            return []
        
        return [self.build]
    
    def _get_run_by_name(self, name: str) -> Run | None:
        """
        Get a run by name or ID from WandB API.
        
        Handles special cases for run identification:
        - "last"    : Returns the most recent run
        - "last[N]" : Returns the Nth most recent run
        - Otherwise : Searches by display_name first, then by run ID
        
        Args:
            name: Run identifier (name, ID, or special syntax)
            
        Returns:
            WandB run object if found, None otherwise
        """
        try:
            if name == "last":
                return runs[0] if (runs := self._api_runs(per_page=1)) else None
            elif name.startswith("last[") and name.endswith("]"):
                n    = int(name[5:-1])
                runs = self._api_runs(per_page=n)
                return runs[n-1] if len(runs) >= n else None
            else:
                if runs := self._api_runs(filters={"display_name": name}):
                    return runs[0]
                try:
                    return self.api.run(f"{self.project}/{name}")
                except Exception:
                    return None
                
        except Exception as e:
            self.ui.print_message(f"Failed to fetch run: {e}", "error")
            return None
    
    def _get_wandb_url(self, run_name: str) -> str | None:
        """
        Get the WandB dashboard URL for a run.
        """
        if run := self._get_run_by_name(run_name):
            return run.url
        return None
    
    def _show_cfg_from_dict(self, cfg: dict[str, Any]):
        """
        Display configuration from a dictionary.
        
        Shared logic for displaying configs from both API runs and dry-run mode.
        Applies build filtering and handles pagination display.
        
        Args:
            cfg : Configuration dictionary with build sections
        """
        cfg = {self.build: cfg.get(self.build, {})} if self.build else cfg
        
        if not (builds := self._get_builds(cfg)):
            return
        
        if not (all_items := [
            item
            for build in builds
            for item in self._flatten_cfg(cfg[build], build).values()
        ]):
            self.ui.print_message(
                message  = "No configuration parameters found",
                msg_type = "info"
            )
            return
        
        self._display_cfg(
            items      = all_items,
            title      = "Configuration Parameters",
            using_last = self.run_id == "last"
        )

    def compare_runs(
        self,
        run1 : str | None = None,
        run2 : str | None = None
    ):
        """
        Compare configurations between two runs.

        Displays side-by-side differences between the configurations of two
        training runs. Can be filtered to show only a specific build.

        The default compares the last two runs. If only `run1` is provided, the
        last run is compared against it.

        Args:
            run1 : First run identifier (defaults to last)
            run2 : Second run identifier (defaults to last[2])
        """
        run1 = run1 or "last"
        run2 = "last" if run1 and not run2 else (run2 or "last[2]")

        if run1 == "last" and run2 == "last[2]":
            self.ui.print_message(
                message  = "Comparing the two most recent runs.",
                msg_type = "info"
            )

        run1_obj = self._ensure_run_exists(run1)
        run2_obj = self._ensure_run_exists(run2)

        cfg1 = self._extract_training_sections(run1_obj.config)
        cfg2 = self._extract_training_sections(run2_obj.config)

        self.ui.print_header("Training Run Management")
        self.ui.print_section("Configuration Comparison", minor=True)
        self.ui.print_message(
            message  = f"Run 1: {run1_obj.name or run1_obj.id}",
            msg_type = "info"
        )
        self.ui.print_message(
            message  = f"Run 2: {run2_obj.name or run2_obj.id}",
            msg_type = "info"
        )

        all_cfgs = cfg1 | cfg2
        domains  = []

        for build in (b for b in self._get_builds(all_cfgs) if b in all_cfgs):
            flat1     = self._flatten_cfg(cfg1.get(build, {}))
            flat2     = self._flatten_cfg(cfg2.get(build, {}))
            get_value = lambda d, k: d.get(k, CfgItem(False, "", "NOT SET")).value

            if differences := [
                (key, val1, val2)
                for key in sorted(set(flat1) | set(flat2))
                if (val1 := get_value(flat1, key)) != (val2 := get_value(flat2, key))
            ]:
                domains.append((build, differences))

        if not domains:
            self.ui.print_message(
                message  = "No configuration differences found between runs",
                msg_type = "info"
            )
            return

        for build, differences in domains:
            table = self.ui.create_aligned_table(
                border_style = "dim",
                columns      = [
                    TableColumn("left", "bright_white", "Parameter", 40),
                    TableColumn("left", "bright_green", "Run 1",     30),
                    TableColumn("left", "bright_blue",  "Run 2",     30)
                ],
                title = f"{build.title()} Configuration"
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
        show_header : bool       = True
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

        try:
            runs = self._api_runs(limit)

        except Exception as e:
            self.ui.print_message(
                message  = f"Failed to fetch runs: {e}",
                msg_type = "error"
            )
            return
            
        if not runs:
            self.ui.print_message(
                message  = "No training runs found. Start training with: "
                           "thermur train",
                msg_type = "info"
            )
            return

        message = (
            f"Showing all {len(runs)} runs."
            if not limit or len(runs) < limit
            else f"Showing {limit} most recent runs. (Use --all to see all runs)"
        )
        
        self.ui.print_message(message, "info")
        self.ui.print_section("Recent Runs", minor=True)
        self.ui.display_status_legend()

        self._display_runs_table(
            [
                TableColumn("left",   "bright_cyan",  "Run ID",    25),
                TableColumn("left",   "bright_white", "Started",   20),
                TableColumn("left",   "bright_green", "Overrides", 45),
                TableColumn("center", "bright_white", "Status",    10)
            ],
            runs
        )

    def show_cfg(self, dry_run: bool = False):
        """
        Display configuration for a run with pagination.

        Shows the full configuration for a training run, optionally
        filtered by build. Large configurations are automatically paginated
        for better readability. In dry-run mode, reads from stdin or shows
        default configuration.
        
        Args:
            dry_run: If True, read config from stdin or show defaults
        """
        try:
            if dry_run:
                if stdin.isatty():
                    cfg       = {
                        k: v for k, v 
                        in app.system.extract_training_cfg(app.cfg).items() 
                        if k != "overrides"
                    }
                    overrides = []
                    run_name  = "Default Configuration"
                else:
                    data      = loads(stdin.read())
                    cfg       = data.get("config", {})
                    overrides = data.get("overrides", [])
                    run_name  = "Dry-Run Configuration"
            else:
                run       = self._ensure_run_exists(self.run_id)
                cfg       = self._extract_training_sections(run.config)
                overrides = run.config.get("overrides", [])
                run_name  = run.name or run.id
                
            self.overrides = overrides
            self.run_name  = run_name
            
        except Exception as e:
            self.ui.print_message(f"Failed to parse cfg: {e}", "error")
            raise Exit(1)
        
        self._show_cfg_from_dict(cfg)
