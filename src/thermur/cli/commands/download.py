"""
Dataset download command for the Thermur CLI.

This module provides the 'download' command for acquiring simulation datasets
from remote repositories. It manages efficient transfers of large-scale NetCDF
files from the Moisseeva (2020) wildfire plume dataset.
"""

from ..helpers import FileIO
from pathlib   import Path
from typer     import Context, Option


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
        self.cfg       = ctx.obj.cfg
        self.cache_dir = Path(self.cfg.file.cache_dir)
        self.file_cfg  = self.cfg.file
        self.prompts   = ctx.obj.prompts
        self.system    = ctx.obj.system
        self.ui        = ctx.obj.ui
        
        # Initialize FileIO with dataset URL
        dataset_url = f"{self.file_cfg.repo_base_url}/{self.file_cfg.dataset_id}"
        self.file_io = FileIO(
            cache_dir   = self.cache_dir,
            chunk_size  = self.file_cfg.chunk_size,
            dataset_url = dataset_url
        )
    
    
    
    
    
    
    def _perform_download(self, file_info: dict):
        """
        Downloads a file via HTTP with progress tracking.
        
        Args:
            file_info: Dictionary with 'name', 'size', and 'url' keys
        """
        self.ui.console.print()
        self.ui.print_minor_section(f"Downloading {file_info['name']}")
        
        # Check resume status
        resume_info = self.file_io.get_resume_info(file_info)
        if resume_info['status'] == 'partial':
            self.ui.print_message(
                f"Resuming from {resume_info['current_size'] / 1e9:.1f} GB "
                f"({resume_info['progress_percent']:.1f}% complete)",
                "info"
            )
        elif resume_info['status'] == 'complete':
            self.ui.print_message(
                f"File already downloaded: {file_info['name']}",
                "success"
            )
            return
        
        # Download with progress
        with self.ui.create_thermal_progress() as progress:
            success = self.file_io.download_file_with_progress(file_info, progress)
        
        self.ui.console.print()
        
        if success:
            if self.file_io.update_manifest(file_info):
                self.ui.print_message(
                    f"Successfully downloaded: {file_info['name']}",
                    "success"
                )
                self.ui.print_message(
                    f"Saved to: {self.cache_dir / file_info['name']}",
                    "info"
                )
            else:
                self.ui.print_message(
                    "Warning: Could not update manifest",
                    "warning"
                )
        else:
            self.ui.print_message(
                "Download failed. Run the command again to resume.",
                "error"
            )
    
    
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
            available_files, existing_files, show_numbers=self.file_cfg.show_numbers_default
        )
        self.ui.console.print()
        
        # File selection and download
        selected_file = self.prompts.select_file_by_number(file_index_map)
        if selected_file and self.prompts.confirm_download(selected_file):
            self._perform_download(selected_file)
        else:
            self.ui.print_message("Download cancelled", "warning")