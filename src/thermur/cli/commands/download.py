"""
Dataset download command for the Thermur CLI.

This module provides the 'download' command for acquiring simulation datasets
from remote repositories. It manages efficient transfers of large-scale NetCDF
files from the Moisseeva (2020) wildfire plume dataset.
"""
from typer import Context, Option


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
    Manages dataset acquisition through Globus transfers.
    
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
        self.globus  = ctx.obj.globus
        self.prompts = ctx.obj.prompts
        self.system  = ctx.obj.system
        self.ui      = ctx.obj.ui
    
    def _perform_download(self, file_info: dict, globus_client):
        """
        Initiates a Globus transfer for the selected file.
        
        Args:
            file_info     : Dictionary with 'name', 'size', and 'path' keys
            globus_client : Authenticated Globus transfer client
        """
        self.ui.console.print()
        self.ui.print_minor_section(f"Preparing transfer for {file_info['name']}")
        
        local_endpoints = self.globus.get_local_endpoints(globus_client)
        
        if not local_endpoints:
            self.ui.print_message(
                "No local Globus endpoint found. Please install and configure Globus Connect Personal.",
                "error"
            )
            return
            
        if len(local_endpoints) == 1:
            local_endpoint = local_endpoints[0]
        else:
            self.ui.print_message("Multiple local endpoints found:", "info")
            for i, ep in enumerate(local_endpoints):
                self.ui.console.print(f"  {i+1}. {ep['display_name']}")
            
            import questionary
            choices = [ep['display_name'] for ep in local_endpoints]
            selected = questionary.select(
                "Select local endpoint:",
                choices=choices
            ).ask()
            
            for ep in local_endpoints:
                if ep['display_name'] == selected:
                    local_endpoint = ep
                    break
        
        dest_path = f"/~/{self.cfg.download.cache_dir}/{file_info['name']}"
        
        self.ui.print_message(f"Source: {self.cfg.download.globus_endpoint_id}", "info")
        self.ui.print_message(f"Destination: {local_endpoint['display_name']}", "info")
        
        try:
            task_id = self.globus.submit_transfer_task(
                source_endpoint = self.cfg.download.globus_endpoint_id,
                dest_endpoint   = local_endpoint['id'],
                items           = [(file_info['path'], dest_path)],
                label           = f"Thermur: {file_info['name']}",
                transfer_client = globus_client
            )
            
            self.ui.print_message(f"Transfer submitted! Task ID: {task_id}", "success")
            self.ui.print_message("You can monitor progress at https://app.globus.org/activity", "info")
            
            if self.prompts.confirm("Wait for transfer to complete?"):
                with self.ui.console.status("[bold green]Transferring file...") as status:
                    success = self.globus.wait_for_transfer(
                        task_id         = task_id,
                        transfer_client = globus_client,
                        timeout         = 3600  # 1 hour timeout
                    )
                
                if success:
                    self.ui.print_message(f"Transfer complete: {file_info['name']}", "success")
                    self.file_io.update_manifest(file_info)
                else:
                    self.ui.print_message("Transfer failed or timed out", "error")
                    
        except Exception as e:
            self.ui.print_message(f"Transfer failed: {str(e)}", "error")
    
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
        
        try:
            globus_client = self.globus.authenticate()
            
            files = self.globus.list_endpoint_directory(
                endpoint_id     = self.cfg.download.globus_endpoint_id,
                path            = self.cfg.download.globus_dataset_path,
                transfer_client = globus_client
            )
            
            available_files = [f for f in files if f["type"] == "file" and f["name"].endswith(".nc")]
            
        except Exception as e:
            self.ui.print_message(
                f"Unable to connect to Globus: {str(e)}",
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
        
        selected_file = self.prompts.select_file_by_number(file_index_map)
        if selected_file and self.prompts.confirm_download(selected_file):
            self._perform_download(selected_file, globus_client)
        else:
            self.ui.print_message("Download cancelled", "warning")