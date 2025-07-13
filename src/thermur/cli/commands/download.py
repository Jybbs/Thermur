"""
Dataset download command for the Thermur CLI.

This module provides the 'download' command for acquiring WRF-Fire simulation
datasets from remote repositories. It manages efficient transfers of large-scale
NetCDF files from the Moisseeva (2020) wildfire plume dataset.
"""
from datetime   import datetime
from pathlib    import Path
from rich.table import Table
from typer      import Context, Option
from typing     import Optional

import json


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
    💾 Download WRF-Fire simulation data for training.
    
    Acquires NetCDF files from configured Globus endpoints, managing large-scale
    transfers efficiently. The system tracks downloaded files in a manifest to
    avoid redundant transfers and enforces size limits for development workflows.
    
    The Moisseeva (2020) dataset contains 147 files totaling 5.33 TB, so subset
    downloads are essential for iterative development.
    """
    command = DownloadCommand(ctx)
    command.run(max_files, max_size, dataset, dry_run)


class DownloadCommand:
    """
    Manages WRF-Fire dataset acquisition through Globus transfers.
    
    Coordinates authentication, transfer initiation, progress tracking, and
    manifest updates. Supports dry-run mode for planning downloads without
    consuming bandwidth or storage.
    """
    
    def __init__(self, ctx: Context):
        """
        Initializes the command with shared context components.
        
        Args:
            ctx: The Typer context containing AppContext with configuration,
                 UI utilities, and system inspection capabilities.
        """
        self.cfg    = ctx.obj.cfg
        self.ui     = ctx.obj.ui
        self.system = ctx.obj.system
        
        # Access dataset configuration
        self.dataset   = self.cfg.dataset
        self.cache_dir = Path(self.dataset.cache_dir)
        self.manifest  = self.cache_dir / "manifest.json"
    
    def run(
        self, 
        max_files : Optional[int], 
        max_size  : Optional[float],
        dataset   : str,
        dry_run   : bool
    ):
        """
        Executes the download workflow.
        
        Args:
            max_files : Override for maximum file count from configuration
            max_size  : Override for maximum total size in GB
            dataset   : Target dataset identifier (e.g., "moisseeva_2020")
            dry_run   : If True, show plan without downloading files
        """
        self.ui.print_header("WRF Data Acquisition")
        
        # Apply CLI overrides to configuration values
        files_limit = max_files if max_files is not None else self.dataset.max_files
        size_limit  = max_size if max_size is not None else self.dataset.max_size_gb
        
        if dry_run:
            self._show_download_plan(dataset, files_limit, size_limit)
            return
            
        self._perform_download(dataset, files_limit, size_limit)
    
    def _show_download_plan(self, dataset: str, max_files: int, max_gb: float):
        """
        Displays the download configuration without executing transfers.
        
        Args:
            dataset   : Dataset name to display in plan
            max_files : Maximum number of files that would be downloaded
            max_gb    : Maximum total size in GB that would be downloaded
        """
        self.ui.print_message("Dry-run mode - no files will be downloaded", "warning")
        
        table = Table(title=f"Download Plan: {dataset}")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Dataset", dataset)
        table.add_row("Endpoint", self.dataset.endpoint_id)
        table.add_row("Max Files", str(max_files))
        table.add_row("Max Size", f"{max_gb:.1f} GB")
        table.add_row("Cache Path", str(self.cache_dir))
        
        self.ui.console.print(table)
        self.ui.console.print()
        self.ui.print_message(
            "Run without --dry-run to execute download",
            "tip"
        )
        
    def _perform_download(self, dataset: str, max_files: int, max_gb: float):
        """
        Performs the actual dataset acquisition through Globus.
        
        Args:
            dataset   : Dataset identifier for manifest tracking
            max_files : Maximum number of files to download
            max_gb    : Maximum total size in GB to download
        """
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            import globus_sdk
            
            with self.ui.console.status("Initializing Globus authentication...", spinner="dots"):
                # Globus OAuth2 flow would be implemented here
                pass
                
            # Placeholder for actual Globus transfer implementation
            self.ui.print_message(
                f"Globus transfer pending - endpoint: {self.dataset.endpoint_id[:8]}...", 
                "error"
            )
            
            # Update manifest to track dataset
            manifest = self._load_manifest()
            if dataset not in manifest:
                manifest[dataset] = {
                    "files"         : [],
                    "total_size_gb" : 0,
                    "last_updated"  : datetime.now().isoformat()
                }
                self._save_manifest(manifest)
                self.ui.print_message(
                    f"Created manifest entry for dataset: {dataset}",
                    "success"
                )
                
        except ImportError:
            self.ui.print_message(
                "Globus SDK not available. Install with: pip install globus-sdk",
                "error"
            )
            self.ui.console.print()
            self.ui.print_message(
                "The Globus SDK is required for efficient large-scale data transfers",
                "tip"
            )
    
    def _load_manifest(self) -> dict:
        """
        Loads the dataset manifest from disk.
        
        Returns:
            Dictionary mapping dataset names to metadata including file lists,
            total sizes, and timestamps. Returns empty dict if manifest doesn't exist.
        """
        if self.manifest.exists():
            with open(self.manifest, 'r') as f:
                return json.load(f)
        return {}
        
    def _save_manifest(self, data: dict):
        """
        Persists the dataset manifest to disk.
        
        Args:
            data: Dictionary containing dataset metadata to save
        """
        with open(self.manifest, 'w') as f:
            json.dump(data, f, indent=2)