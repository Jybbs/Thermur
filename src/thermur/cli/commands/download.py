"""
Dataset download command for the Thermur CLI.

This module provides the 'download' command for acquiring simulation datasets
from remote repositories. It manages efficient transfers of large-scale NetCDF
files from the Moisseeva (2020) wildfire plume dataset.
"""
from typer import Context, Option

import requests


def download(
    ctx  : Context,
    list : bool = Option(
        False,
        "--list", "-l", 
        help = "List available files and their download status"
    )
):
    """
    📥 Download simulation data for training.
    
    Downloads NetCDF files from the Moisseeva (2020) wildfire plume dataset
    hosted on FRDR. The system tracks downloaded files and shows checkmarks
    for files you already have. Interactive file selection allows you to choose
    exactly which file to download.
    
    The dataset contains 147 LES simulations totaling 5.33 TB, with individual
    files ranging from 20-50 GB each. Each file represents a different fire
    scenario with varying conditions (case, fire type, run number).
    
    Examples:
        thermur download --list    # Show all files with download status
        thermur download           # Interactive selection and download
    """
    command = DownloadCommand(ctx)
    command.run(list)


class DownloadCommand:
    """
    Manages dataset acquisition through HTTP transfers.
    
    Coordinates file listing, selection, download progress tracking, and
    manifest updates. Downloads individual files from the FRDR repository.
    """
    
    def __init__(self, ctx: Context):
        """
        Initializes the command with shared context components.
        
        Args:
            ctx: The Typer context containing AppContext with configuration,
                 UI utilities, and system inspection capabilities.
        """
        self.cfg     = ctx.obj.cfg
        self.file_io = ctx.obj.file_io
        self.prompts = ctx.obj.prompts
        self.system  = ctx.obj.system
        self.ui      = ctx.obj.ui
    
    def _perform_download(self, file_info: dict):
        """
        Downloads a file via HTTP with progress tracking.
        
        Args:
            file_info: Dictionary with 'name', 'size', and 'url' keys
        """
        self.ui.console.print()
        self.ui.print_minor_section(f"Downloading {file_info['name']}")
        
        resume = self.file_io.get_resume_info(file_info)
        if resume['status'] == 'partial':
            gb  = resume['current_size'] / 1e9
            pct = resume['progress_percent']
            self.ui.print_message(f"Resuming from {gb:.1f} GB ({pct:.1f}%)", "info")
        
        # Download with progress
        try:
            with self.ui.create_thermal_progress() as progress:
                task = progress.add_task(
                    completed   = resume['current_size'],
                    description = f"[cyan]{file_info['name']}[/cyan]",
                    total       = file_info['size']
                )
                
                for bytes_down, status in self.file_io.download_chunks(file_info):
                    if status == 'complete':
                        self.ui.print_message(
                            message  = f"Already downloaded: {file_info['name']}", 
                            msg_type = "success"
                        )
                        return
                    progress.update(task, advance=bytes_down)
            
            self.ui.console.print()
            if self.file_io.update_manifest(file_info):
                self.ui.print_message(f"Downloaded: {file_info['name']}", "success")
                path = self.cfg.download.cache_dir / file_info['name']
                self.ui.print_message(f"Saved to: {path}", "info")
            else:
                self.ui.print_message("Could not update manifest", "warning")
                
        except requests.exceptions.RequestException:
            self.ui.console.print()
            self.ui.print_message("Download failed. Run again to resume.", "error")
    
    def _show_files_and_summary(
        self,
        available_files : list[dict],
        existing_files  : set[str],
        show_numbers    : bool = False,
    ) -> dict[int, dict]:
        """
        Display file table and summary.
        
        Args:
            available_files : All available files
            existing_files  : Already downloaded files
            show_numbers    : Whether to show selection numbers
            
        Returns:
            File index mapping if show_numbers is True
        """
        table, file_index_map = self.ui.create_file_table(
            available_files  = available_files,
            existing_files   = existing_files,
            group_extractor  = lambda name: name.split('F')[0] if 'F' in name else "Unknown",
            show_numbers     = show_numbers,
            title            = "Moisseeva (2020) Dataset Files"
        )
        self.ui.console.print(table)
        self.ui.console.print()
        self.ui.display_file_summary(available_files, existing_files)
        
        return file_index_map if show_numbers else {}
    
    def run(self, list_only: bool):
        """
        Executes the download workflow.
        
        Args:
            list_only: If True, only list files without downloading
        """
        self.ui.print_header("Data Acquisition")
        
        # Get file listings
        self.ui.print_message(
            "Note: Using representative file listing. FRDR integration pending dataset availability.",
            "warning"
        )
        
        available_files = self.file_io.fetch_file_listing()
        if not available_files:
            self.ui.print_message(
                "Unable to fetch file listing. The dataset may be temporarily unavailable.",
                "error"
            )
            return
            
        existing_files = self.file_io.check_existing_files()
        
        # List mode - show all files with status
        if list_only:
            self._show_files_and_summary(available_files, existing_files, show_numbers=False)
            return
            
        # Download mode - check if anything to download
        files_to_download = self.file_io.get_undownloaded_files(available_files)
        
        if not files_to_download:
            self.ui.print_message("All files already downloaded! 🎉", "success")
            return
            
        # Show files with selection numbers
        file_index_map = self._show_files_and_summary(
            available_files, existing_files, show_numbers=self.cfg.download.show_numbers_default
        )
        self.ui.console.print()
        
        # File selection and download
        selected_file = self.prompts.select_file_by_number(file_index_map)
        if selected_file and self.prompts.confirm_download(selected_file):
            self._perform_download(selected_file)
        else:
            self.ui.print_message("Download cancelled", "warning")