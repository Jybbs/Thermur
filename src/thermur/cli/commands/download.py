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
        "--max-files",
        help = "Override maximum number of files to download"
    ),
    dataset   : str = Option(
        "moisseeva_2020",
        "--dataset",
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
    
    Acquires NetCDF files from configured Globus endpoints, managing large-scale
    transfers efficiently. The system tracks downloaded files in a manifest to
    avoid redundant transfers and checks for existing files before downloading.
    
    The Moisseeva (2020) dataset contains 147 files totaling 5.33 TB, with
    individual files ranging from 20-50 GB each.
    """
    command = DownloadCommand(ctx)
    command.run(max_files, dataset, dry_run)


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
        self.cfg       = ctx.obj.cfg
        self.dataset   = ctx.obj.cfg.dataset
        self.max_files = self.dataset.max_files
        self.prompts   = ctx.obj.prompts
        self.system    = ctx.obj.system
        self.ui        = ctx.obj.ui
        
        self.cache_dir = Path(self.dataset.cache_dir)
        self.manifest  = self.cache_dir / "manifest.json"
    
    def run(
        self, 
        max_files : Optional[int],
        dataset   : str,
        dry_run   : bool
    ):
        """
        Executes the download workflow.
        
        Args:
            max_files : CLI override for maximum file count (None uses config value)
            dataset   : Target dataset identifier
            dry_run   : If True, show plan without downloading files
        """
        self.ui.print_header("WRF Data Acquisition")
        
        if max_files is not None:
            self.max_files = max_files
        
        existing_files = self._check_existing_files(dataset)
        files_needed   = max(0, self.max_files - len(existing_files))
        
        self._show_download_plan(dataset, existing_files, files_needed, dry_run)
        
        if files_needed == 0:
            return
            
        if not dry_run and self._confirm_download(files_needed):
            self._perform_download(dataset)
    
    def _check_existing_files(self, dataset: str) -> list[str]:
        """
        Checks manifest for already downloaded files.
        
        Args:
            dataset: Dataset name to check
            
        Returns:
            List of filenames that have already been downloaded
        """
        manifest = self._load_manifest()
        if dataset in manifest:
            return manifest[dataset].get("files", [])
        return []
    
    def _confirm_download(self, files_needed: int) -> bool:
        """
        Prompts user to confirm the download operation.
        
        Args:
            files_needed: Number of files that will be downloaded
            
        Returns:
            True if user confirms, False otherwise
        """
        estimated_size = files_needed * 35
        return self.prompts.confirm_download(files_needed, estimated_size)
    
    def _load_manifest(self) -> dict:
        """
        Loads the dataset manifest from disk.
        
        Returns:
            Dictionary mapping dataset names to metadata including file lists
            and timestamps. Returns empty dict if manifest doesn't exist.
        """
        if self.manifest.exists():
            with open(self.manifest, 'r') as f:
                return json.load(f)
        return {}
    
    def _perform_download(self, dataset: str):
        """
        Performs the actual dataset acquisition through Globus.
        
        Args:
            dataset: Dataset identifier for manifest tracking
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            import globus_sdk
            
            with self.ui.console.status("Initializing Globus authentication...", spinner="dots"):
                # TODO: Implement Globus OAuth2 flow
                pass
            
            self.ui.print_message(
                f"Globus transfer pending - endpoint: {self.dataset.endpoint_id[:8]}...", 
                "error"
            )
            
            manifest = self._load_manifest()
            if dataset not in manifest:
                manifest[dataset] = {
                    "files"        : [],
                    "last_updated" : datetime.now().isoformat()
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
    
    def _save_manifest(self, data: dict):
        """
        Persists the dataset manifest to disk.
        
        Args:
            data: Dictionary containing dataset metadata to save
        """
        with open(self.manifest, 'w') as f:
            json.dump(data, f, indent=2)
            
    def _show_download_plan(
        self,
        dataset        : str,
        existing_files : list[str], 
        files_needed   : int,
        dry_run        : bool
    ):
        """
        Displays the download configuration.
        
        Args:
            dataset        : Dataset name to display in plan
            existing_files : List of already downloaded files
            files_needed   : Number of new files to download
            dry_run        : Whether this is a dry-run
        """
        if dry_run:
            self.ui.print_message("Dry-run mode - no files will be downloaded", "warning")
        
        table = Table(title=f"Download Plan: {dataset}")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value",     style="green")
        
        table.add_row("Dataset",           dataset)
        table.add_row("Endpoint",          self.dataset.endpoint_id)
        table.add_row("Max Files",         str(self.max_files))
        table.add_row("Existing Files",    str(len(existing_files)))
        table.add_row("Files to Download", str(files_needed))
        table.add_row("Cache Path",        str(self.cache_dir))
        
        self.ui.console.print(table)
        self.ui.console.print()
        
        if files_needed > 0:
            verb = "Would download" if dry_run else "Will download"
            self.ui.print_message(
                f"{verb} {files_needed} new file(s) (~{files_needed * 35} GB)",
                "info"
            )
            if dry_run:
                self.ui.print_message(
                    "Run without --dry-run to execute download",
                    "tip"
                )
        else:
            self.ui.print_message(
                "All requested files already downloaded",
                "success"
            )