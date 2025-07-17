"""
Dataset download command for the Thermur CLI.

This module provides the 'download' command for acquiring simulation datasets
from remote repositories. It manages efficient transfers of large-scale NetCDF
files from the Moisseeva (2020) wildfire plume dataset.
"""
from globus_sdk import TransferClient
from typer      import Context


def download(ctx: Context):
    """
    📥 Download simulation data for training.
    
    Shows NetCDF files from the Moisseeva (2020) wildfire plume dataset
    hosted on FRDR. Files you already have are marked with checkmarks.
    Select any file to download or re-download.
    
    The dataset contains 147 LES simulations totaling 5.33 TB, with individual
    files ranging from 20-50 GB each. Each file represents a different fire
    scenario with varying conditions (case, fire type, run number).
    
    Example:
        thermur download    # Show files and select for download
    """
    command = DownloadCommand(ctx)
    command.run()


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
        self.cache_dir = ctx.obj.cfg.download.cache_dir
        self.cfg       = ctx.obj.cfg
        self.globus    = ctx.obj.globus
        self.prompts   = ctx.obj.prompts
        self.system    = ctx.obj.system
        self.ui        = ctx.obj.ui
    
    def _get_available_files(self, globus_client: TransferClient) -> list[dict]:
        """
        Retrieve list of NetCDF files from the Globus endpoint.
        
        Args:
            globus_client : Authenticated transfer client
            
        Returns:
            List of file dictionaries with 'name', 'size', and 'path' keys
        """
        files = self.globus.list_endpoint_directory(
            endpoint_id     = self.cfg.download.globus_endpoint_id,
            path            = self.cfg.download.globus_dataset_path,
            transfer_client = globus_client
        )
        
        return [
            f for f in files 
            if f["type"] == "file" and f["name"].endswith(".nc")
        ]
    
    def _get_download_status(self, available_files: list[dict]) -> dict[str, str]:
        """
        Check download status for each file.
        
        Compares available files against local cache directory to determine
        status: 'downloaded', 'incomplete', or 'missing'.
        
        Args:
            available_files : List of files from Globus with size info
            
        Returns:
            Dict mapping filename to status
        """
        if not self.cache_dir.exists():
            return {f['name']: 'missing' for f in available_files}
            
        status = {}
        for file_info in available_files:
            local_path = self.cache_dir / file_info['name']
            
            if not local_path.exists():
                status[file_info['name']] = 'missing'
            elif local_path.stat().st_size != file_info['size']:
                status[file_info['name']] = 'incomplete'
            else:
                status[file_info['name']] = 'downloaded'
                
        return status
    
    def _get_local_endpoint(self, globus_client: TransferClient) -> dict | None:
        """
        Retrieve and select a local Globus endpoint for transfers.
        
        Queries for available local Globus Connect Personal endpoints and
        prompts the user to select one if multiple are found. Returns None
        if no endpoints are available or selection is cancelled.
        
        Args:
            globus_client : Authenticated Globus transfer client
            
        Returns:
            Selected endpoint dictionary or None if unavailable
        """
        local_endpoints = self.globus.get_local_endpoints(globus_client)
        
        if not local_endpoints:
            self.ui.print_message(
                message  = (
                    "No local Globus endpoint found. "
                    "Please install and configure Globus Connect Personal."
                ),
                msg_type = "error"
            )
            return None
            
        selected = self.prompts.select_globus_endpoint(local_endpoints)
        if not selected:
            self.ui.print_message("No endpoint selected", "warning")
            
        return selected
    
    def _initiate_transfer(
        self,
        dest_endpoint   : str,
        file_info       : dict,
        globus_client   : TransferClient,
        source_endpoint : str
    ) -> str | None:
        """
        Submit a file transfer task to Globus.
        
        Creates and submits a transfer task for a single file. Returns the
        task ID for monitoring or None if submission fails.
        
        Args:
            dest_endpoint   : UUID of the destination endpoint
            file_info       : File metadata including path and name
            globus_client   : Authenticated transfer client
            source_endpoint : UUID of the source endpoint
            
        Returns:
            Task ID string or None if submission failed
        """
        dest_path = f"/~/{self.cfg.download.cache_dir}/{file_info['name']}"
        
        try:
            task_id = self.globus.submit_transfer_task(
                dest_endpoint   = dest_endpoint,
                items           = [(file_info['path'], dest_path)],
                label           = f"Thermur: {file_info['name']}",
                source_endpoint = source_endpoint,
                transfer_client = globus_client
            )
            return task_id
        
        except Exception as e:
            self.ui.print_message(f"Transfer submission failed: {str(e)}", "error")
            return None
    
    def _monitor_transfer(
        self,
        file_info     : dict,
        globus_client : TransferClient,
        task_id       : str
    ) -> bool:
        """
        Monitor a transfer task until completion or timeout.
        
        Displays a progress indicator while waiting for the transfer to
        complete. Updates the manifest on success. Returns True if the
        transfer completed successfully.
        
        Args:
            file_info     : File metadata for manifest update
            globus_client : Authenticated transfer client
            task_id       : ID of the transfer task to monitor
            
        Returns:
            True if transfer succeeded, False otherwise
        """
        with self.ui.console.status("[bold green]Transferring file..."):
            success = self.globus.wait_for_transfer(
                task_id         = task_id,
                transfer_client = globus_client,
                timeout         = 3600 
            )
        
        if success:
            self.ui.print_message(f"Transfer complete: {file_info['name']}", "success")
        else:
            self.ui.print_message("Transfer failed or timed out", "error")
            
        return success
    
    def _handle_file_selection(
        self,
        file_info     : dict,
        file_status   : dict[str, str],
        globus_client : TransferClient
    ) -> None:
        """
        Handle user's file selection with appropriate prompts.
        
        Checks the file's current status and prompts for confirmation
        if needed before initiating the download.
        
        Args:
            file_info     : Selected file information
            file_status   : Dict mapping filenames to status
            globus_client : Authenticated transfer client
        """
        status = file_status.get(file_info['name'], 'missing')
        
        if status == 'downloaded':
            if not self.prompts.confirm(f"{file_info['name']} is already downloaded. Re-download?"):
                return
        elif status == 'incomplete':
            self.ui.print_message(
                f"{file_info['name']} appears incomplete. Will re-download.",
                "warning"
            )
                
        if self.prompts.confirm_download(file_info):
            self._perform_download(file_info, globus_client)
    
    def _perform_download(self, file_info: dict, globus_client: TransferClient):
        """
        Orchestrate the complete download workflow for a single file.
        
        Coordinates endpoint selection, transfer initiation, and optional
        monitoring. This is the main entry point for downloading a file
        after user selection.
        
        Args:
            file_info     : Dictionary with 'name', 'size', and 'path' keys
            globus_client : Authenticated Globus transfer client
        """
        self.ui.console.print()
        self.ui.print_minor_section(f"Preparing transfer for {file_info['name']}")
        
        local_endpoint = self._get_local_endpoint(globus_client)
        if not local_endpoint:
            return
        
        self.ui.print_message(
            message  = f"Source: {self.cfg.download.globus_endpoint_id}", 
            msg_type = "info"
        )
        self.ui.print_message(
            message  = f"Destination: {local_endpoint['display_name']}", 
            msg_type = "info"
        )
        
        task_id = self._initiate_transfer(
            dest_endpoint   = local_endpoint['id'],
            file_info       = file_info,
            globus_client   = globus_client,
            source_endpoint = self.cfg.download.globus_endpoint_id
        )
        
        if not task_id:
            return
            
        self.ui.print_message(f"Transfer submitted! Task ID: {task_id}", "success")
        self.ui.print_message("You can monitor progress at https://app.globus.org/activity", "info")
        
        if self.prompts.confirm("Wait for transfer to complete?"):
            self._monitor_transfer(file_info, globus_client, task_id)
    
    
    def run(self):
        """
        Executes the download workflow.
        
        Shows all available files with status indicators (downloaded, 
        incomplete, or missing) and allows selection for download.
        """
        self.ui.print_header("Data Acquisition")
        
        try:
            globus_client   = self.globus.authenticate()
            available_files = self._get_available_files(globus_client)
        except Exception as e:
            self.ui.print_message(f"Unable to connect to Globus: {str(e)}", "error")
            return
            
        file_status = self._get_download_status(available_files)
        
        file_index_map = self.ui.display_download_table(
            available_files = available_files,
            file_status     = file_status,
            title           = "Moisseeva (2020) Dataset Files"
        )
        
        self.ui.console.print()
        self.ui.display_download_summary(available_files, file_status)
        self.ui.console.print()
        
        selected_file = self.prompts.select_file_by_number(file_index_map)
        if selected_file:
            self._handle_file_selection(selected_file, file_status, globus_client)
        else:
            self.ui.print_message("Download cancelled", "warning")