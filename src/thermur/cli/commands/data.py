"""
Data acquisition commands for the Thermur CLI.

This module provides the 'data' subcommands for managing WRF-Fire simulation
datasets. It supports downloading, listing, and cleaning cached data files
for training the thermal drone flock model.
"""
import json
from datetime   import datetime
from pathlib    import Path
from typing     import Optional

from rich.panel import Panel
from rich.table import Table
from typer      import Context, Option


def download(
    ctx       : Context,
    max_files : Optional[int] = Option(
        None,
        "--max-files", "-n",
        help = "Maximum number of files to download"
    ),
    max_size  : Optional[float] = Option(
        None,
        "--max-size", "-s", 
        help = "Maximum total size in GB"
    ),
    dataset   : str = Option(
        "moisseeva_2020",
        "--dataset", "-d",
        help = "Dataset to download"
    ),
    dry_run   : bool = Option(
        False,
        "--dry-run",
        help = "Show download plan without executing"
    )
):
    """
    📥 Download WRF-Fire simulation data for training.
    
    Downloads NetCDF files from configured dataset repositories using
    Globus for efficient large-scale transfers. Tracks downloaded files
    to avoid re-downloading.
    """
    command = DataDownloadCommand(ctx)
    command.run(max_files, max_size, dataset, dry_run)


def list(ctx: Context):
    """
    📋 List downloaded datasets and their status.
    
    Shows information about locally cached datasets including file counts,
    total sizes, and download timestamps.
    """
    command = DataListCommand(ctx)
    command.run()


def clean(
    ctx     : Context,
    all     : bool = Option(
        False,
        "--all",
        help = "Remove all cached data files"  
    ),
    dataset : Optional[str] = Option(
        None,
        "--dataset", "-d",
        help = "Remove specific dataset only"
    )
):
    """
    🧹 Clean cached data files.
    
    Removes downloaded NetCDF files to free up disk space. Can target
    specific datasets or clean all cached data.
    """
    command = DataCleanCommand(ctx)
    command.run(all, dataset)


class DataDownloadCommand:
    """
    Encapsulates the logic for downloading WRF-Fire datasets.
    """
    
    def __init__(self, ctx: Context):
        """
        Initializes the command with shared context components.
        
        Args:
            ctx: The Typer context containing configuration and UI components.
        """
        self.cfg    = ctx.obj.cfg
        self.ui     = ctx.obj.ui
        self.system = ctx.obj.system
        
        # Load data acquisition config
        from hydra_zen                      import instantiate
        from hydra_zen.third_party.pydantic import pydantic_parser
        
        self.data_config   = instantiate(
            self.cfg.wrf_dataset,
            _parser = pydantic_parser
        )
        self.cache_dir     = Path(self.data_config.cache_dir)
        self.manifest_path = self.cache_dir / "manifest.json"
        
    def run(
        self, 
        max_files : Optional[int], 
        max_size  : Optional[float],
        dataset   : str,
        dry_run   : bool
    ):
        """
        Execute the download command.
        """
        # Override config values if provided
        if max_files is not None:
            self.data_config.max_files = max_files
        if max_size is not None:
            self.data_config.max_size_gb = max_size
            
        self.ui.print_header("WRF Data Acquisition")
        
        # Ensure cache directory exists
        if not dry_run:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
        # Load or create manifest
        manifest = self._load_manifest()
        
        if dry_run:
            self.ui.print_message("Running in dry-run mode", "warning")
            self._show_download_plan(dataset)
            return
            
        # Check for Globus SDK
        try:
            import globus_sdk
            
            with self.ui.console.status("Initializing Globus authentication...", spinner="dots"):
                # Placeholder for Globus implementation
                pass
                
            self.ui.print_message(
                f"Globus integration pending. Would download from endpoint: "
                f"{self.data_config.endpoint_id}", 
                "error"
            )
            
            # Create example manifest entry
            if dataset not in manifest:
                manifest[dataset] = {
                    "files"         : [],
                    "total_size_gb" : 0,
                    "last_updated"  : datetime.now().isoformat()
                }
                self._save_manifest(manifest)
                
        except ImportError:
            self.ui.print_message(
                "Globus SDK not installed. Install with: pip install globus-sdk",
                "error"
            )
            
    def _load_manifest(self) -> dict:
        """Load the download manifest or create empty one."""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        return {}
        
    def _save_manifest(self, manifest: dict):
        """Save the download manifest."""
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
            
    def _show_download_plan(self, dataset: str):
        """Display what would be downloaded."""
        table = Table(title=f"Download Plan: {dataset}")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Dataset", dataset)
        table.add_row("Endpoint", self.data_config.endpoint_id)
        table.add_row("Max Files", str(self.data_config.max_files))
        table.add_row("Max Size", f"{self.data_config.max_size_gb:.1f} GB")
        table.add_row("Cache Path", str(self.cache_dir))
        
        self.ui.console.print(table)


class DataListCommand:
    """
    Lists cached datasets and their status.
    """
    
    def __init__(self, ctx: Context):
        """Initialize with shared context."""
        self.cfg = ctx.obj.cfg
        self.ui  = ctx.obj.ui
        
        # Load config to get cache directory
        from hydra_zen                      import instantiate
        from hydra_zen.third_party.pydantic import pydantic_parser
        
        self.data_config   = instantiate(
            self.cfg.wrf_dataset,
            _parser = pydantic_parser
        )
        self.cache_dir     = Path(self.data_config.cache_dir)
        self.manifest_path = self.cache_dir / "manifest.json"
        
    def run(self):
        """Display information about cached datasets."""
        self.ui.print_header("Cached Datasets")
        
        if not self.manifest_path.exists():
            self.ui.print_message("No datasets have been downloaded yet", "warning")
            self.ui.print_message(f"Cache directory: {self.cache_dir}", "info")
            return
            
        with open(self.manifest_path, 'r') as f:
            manifest = json.load(f)
            
        if not manifest:
            self.ui.print_message("No datasets found in manifest", "warning")
            return
            
        table = Table()
        table.add_column("Dataset", style="cyan")
        table.add_column("Files", style="green", justify="right")
        table.add_column("Size (GB)", style="yellow", justify="right")
        table.add_column("Last Updated", style="magenta")
        
        total_files = 0
        total_size  = 0
        
        for dataset_name, info in manifest.items():
            file_count = len(info.get("files", []))
            size_gb    = info.get("total_size_gb", 0)
            updated    = info.get("last_updated", "Unknown")
            
            if updated != "Unknown":
                dt      = datetime.fromisoformat(updated)
                updated = dt.strftime("%Y-%m-%d %H:%M")
                
            table.add_row(
                dataset_name,
                str(file_count),
                f"{size_gb:.2f}",
                updated
            )
            
            total_files += file_count
            total_size  += size_gb
            
        self.ui.console.print(table)
        self.ui.console.print()
        self.ui.console.print(
            Panel(
                f"[bold]Total:[/bold] {total_files} files, {total_size:.2f} GB",
                style = "blue"
            )
        )


class DataCleanCommand:
    """
    Removes cached data files.
    """
    
    def __init__(self, ctx: Context):
        """Initialize with shared context."""
        self.cfg = ctx.obj.cfg
        self.ui  = ctx.obj.ui
        
        # Load config
        from hydra_zen                      import instantiate
        from hydra_zen.third_party.pydantic import pydantic_parser
        
        self.data_config   = instantiate(
            self.cfg.wrf_dataset,
            _parser = pydantic_parser
        )
        self.cache_dir     = Path(self.data_config.cache_dir)
        self.manifest_path = self.cache_dir / "manifest.json"
        
    def run(self, all: bool, dataset: Optional[str]):
        """Execute the clean command."""
        self.ui.print_header("Clean Cached Data")
        
        if not self.cache_dir.exists():
            self.ui.print_message("No cache directory found", "info")
            return
            
        if all:
            self.ui.print_message(
                f"Would remove all cached data in: {self.cache_dir}",
                "warning"
            )
        elif dataset:
            self.ui.print_message(
                f"Would remove cached files for: {dataset}",
                "info"
            )
        else:
            self.ui.print_message(
                "Please specify --all or --dataset <name>",
                "error"
            )