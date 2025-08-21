"""
Dataset download command for the Thermur CLI.

This module provides the 'download' command for acquiring simulation datasets
from remote repositories. It manages efficient transfers of large-scale NetCDF
files from the Moisseeva (2020) wildfire plume dataset.
"""
from __future__  import annotations
from itertools   import accumulate
from pathlib     import Path
from requests    import get
from tarfile     import open as tar_open
from tempfile    import NamedTemporaryFile
from thermur.cli import app
from time        import perf_counter
from typer       import Exit, Option
from typing      import IO, TYPE_CHECKING
from webbrowser  import open as web_open

if TYPE_CHECKING:
    from config.types  import FileInfo, TransferStatus
    from globus_sdk    import TransferClient
    from requests      import Response
    from rich.progress import Progress, TaskID


def download(
    sample    : bool = Option(
        False,
        "--sample", "-s",
        help = "Download sample dataset (468MB compressed, 1.5GB extracted)"
    ),
    wrf_sfire : bool = Option(
        False,
        "--wrf-sfire", "-w",
        help = "Browse full FRDR dataset (5.3TB, 147 files)"
    )
):
    """
    📥 Download simulation data for training.

    Choose between a curated sample dataset (468MB compressed, 1.5GB extracted)
    hosted on Hugging Face, or browse the full Moisseeva (2020) wildfire plume
    dataset (5.3TB total) hosted on FRDR via Globus.

    The full dataset contains 147 LES simulations with individual files ranging
    from 20-50GB each. Each file represents a different fire scenario with
    varying conditions (wind speed, fuel type, atmospheric profile).

    Examples:
        thermur download              # Interactive mode - choose data source
        thermur download --sample     # Download sample dataset directly
        thermur download --wrf-sfire  # Browse full FRDR dataset
        thermur download -w           # Short form for wrf-sfire
    """
    if sample and wrf_sfire:
        app.ui.print_message(
            message  = "Cannot specify both --sample and --wrf-sfire",
            msg_type = "error"
        )
        raise Exit(1)

    command = DownloadCommand()
    command.run(sample, wrf_sfire)


class DownloadCommand:
    """
    Manages dataset acquisition through Globus transfers.

    Coordinates file listing, selection, download progress tracking, and
    manifest updates. Downloads individual files from the FRDR repository.
    """

    def __init__(self):
        """
        Initializes the command with shared context components.
        """
        self.cfg           = app.cfg
        self.globus        = app.get_globus()
        self.prompts       = app.prompts
        self.system        = app.system
        self.ui            = app.ui
        self.wrf_sfire_dir = Path(app.cfg.download.wrf_sfire_dir)
    
    def _download_sample(self):
        """
        Downloads and extracts the sample data from Hugging Face.

        Downloads a tar.gz file containing sample WRF data and extracts it
        to the data/raw/sample directory.
        """
        sample_file = Path(self.cfg.download.sample_data_path)

        if sample_file.exists() and not self.prompts.confirm(
            "Sample data exists. Re-download?"
        ):
            self.ui.print_message(f"Using existing sample at {sample_file}", "info")
            return

        sample_file.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = None
        try:
            response = get(self.cfg.download.sample_data_url, stream=True)
            response.raise_for_status()

            with NamedTemporaryFile(delete=False, suffix='.tar.gz') as tmp:
                tmp_path = Path(tmp.name)
                self._stream_http_download(
                    filename = "sample data (468 MB)",
                    output   = tmp,
                    response = response,
                    size     = int(response.headers.get('content-length', 0))
                )

            self.ui.print_message("Extracting sample data...", "info")

            with tar_open(tmp_path, 'r:gz') as tar:
                tar.extractall(self.cfg.download.sample_extract_dir)

            self.ui.print_message(
                message  = f"Sample data ready at {sample_file}",
                msg_type = "success"
            )

        except Exception as e:
            self.ui.print_message(
                message  = f"Failed to download sample data: {str(e)}",
                msg_type = "error"
            )

        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()

    def _ensure_authentication(self) -> TransferClient:
        """
        Ensure we have an authenticated Globus client.

        Attempts to use existing authentication tokens, and if not available,
        guides the user through the browser-based OAuth2 flow.

        Returns:
            Authenticated TransferClient

        Raises:
            Exception: If authentication fails
        """
        globus_client = self.globus.get_or_create_client()

        if globus_client is not None:
            return globus_client

        auth_client, auth_url = self.globus.start_oauth2_flow()
        self.ui.print_auth_prompt(auth_url)

        if self.prompts.confirm("Open browser to complete authentication?"):
            try:
                web_open(auth_url, new=2)
                self.ui.print_message(
                    message  = "Browser opened successfully",
                    msg_type = "success"
                )
            except Exception:
                self.ui.print_message(
                    message  = "Unable to open browser automatically",
                    msg_type = "warning"
                )

        if auth_code := input("Enter the authorization code from the browser: "):
            globus_client = self.globus.finalize_oauth2_flow(
                auth_code = auth_code,
                client    = auth_client
            )
            self.ui.print_message(
                message  = "Authentication successful! Credentials saved.",
                msg_type = "success"
            )
            return globus_client
        else:
            raise Exception("Authentication cancelled by user")

    def _get_available_files(self, globus_client: TransferClient) -> list[FileInfo]:
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
            if f["type"] == "file" and f["name"].startswith("wrfout_")
        ]

    def _get_download_status(self, available_files: list[FileInfo]) -> dict[str, str]:
        """
        Check download status for each file.

        Compares available files against local cache directory to determine
        status: 'downloaded', 'incomplete', or 'missing'.

        Args:
            available_files : List of files from Globus with size info

        Returns:
            Dict mapping filename to status
        """
        if not self.wrf_sfire_dir.exists():
            return {f['name']: 'missing' for f in available_files}

        return {
            file_info['name']: (
                'downloaded' if (
                    (local_path := self.wrf_sfire_dir / file_info['name']).exists()
                    and local_path.stat().st_size == file_info['size']
                )
                else 'incomplete' if local_path.exists()
                else 'missing'
            )
            for file_info in available_files
        }

    def _handle_file_selection(
        self,
        file_info     : FileInfo,
        file_status   : dict[str, str],
        globus_client : TransferClient
    ):
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

        match status:
            case 'downloaded':
                message = f"{file_info['name']} is already downloaded. Re-download?"
                if not self.prompts.confirm(message):
                    return
            case 'incomplete':
                self.ui.print_message(
                    f"{file_info['name']} appears incomplete. Will re-download.",
                    "warning"
                )
            case _:
                pass

        if self.prompts.confirm_download(file_info):
            self._perform_download(file_info, globus_client)

    def _initiate_transfer(
        self,
        dest_endpoint   : str,
        file_info       : FileInfo,
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
        dest_path = Path("/~") / self.cfg.download.wrf_sfire_dir / file_info['name']

        try:
            task_id = self.globus.submit_transfer_task(
                dest_endpoint   = dest_endpoint,
                items           = [(file_info['path'], str(dest_path))],
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
        file_info     : FileInfo,
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
        with self.ui.create_thermal_progress() as progress:
            task = progress.add_task(
                description = f"Downloading {file_info['name']}",
                total       = file_info['size']
            )

            def update_callback(status: TransferStatus):
                self._update_progress(
                    progress   = progress,
                    task       = task,
                    filename   = file_info['name'],
                    bytes_done = status["bytes_transferred"],
                    rate_mbps  = status["mbps"],
                    status     = status["nice_status"]
                )

            success = self.globus.wait_for_transfer(
                progress_callback = update_callback,
                task_id           = task_id,
                timeout           = self.cfg.download.transfer_timeout,
                transfer_client   = globus_client
            )

        if success:
            self.ui.print_message(f"Transfer complete: {file_info['name']}", "success")
        else:
            self.ui.print_message("Transfer failed or timed out", "error")

        return success

    def _perform_download(
        self,
        file_info     : FileInfo,
        globus_client : TransferClient
    ):
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
        self.ui.print_section(
            minor = True,
            title = f"Preparing transfer for {file_info['name']}"
        )

        local_endpoint = self.globus.get_local_endpoints(globus_client)[0]

        if not local_endpoint:
            self.ui.print_message(
                message  = (
                    "No local Globus endpoint found. "
                    "Please install and configure Globus Connect Personal."
                ),
                msg_type = "error"
            )
            return

        self.ui.print_message(
            message  = f"Source: {self.cfg.download.globus_endpoint_id}",
            msg_type = "info"
        )
        self.ui.print_message(
            message  = f"Destination: {local_endpoint['display_name']}",
            msg_type = "info"
        )

        if not (task_id :=
            self._initiate_transfer(
                dest_endpoint   = local_endpoint['id'],
                file_info       = file_info,
                globus_client   = globus_client,
                source_endpoint = self.cfg.download.globus_endpoint_id
            )
        ):
            return

        self.ui.print_message(f"Transfer submitted! Task ID: {task_id}", "success")
        self.ui.print_message(
            message  = "You can monitor progress at https://app.globus.org/activity",
            msg_type = "info"
        )

        if self.prompts.confirm("Wait for transfer to complete?"):
            self._monitor_transfer(file_info, globus_client, task_id)


    def _stream_http_download(
        self,
        filename : str,
        output   : IO[bytes],
        response : Response,
        size     : int
    ):
        """
        Stream HTTP download with progress tracking.

        Args:
            filename : Display name for progress bar
            output   : File object to write to
            response : requests Response object with stream=True
            size     : Total size in bytes
        """
        with self.ui.create_thermal_progress() as progress:
            task = progress.add_task(
                description = f"Downloading {filename}",
                total       = size
            )
            start = perf_counter()

            for downloaded in accumulate(
                len(chunk) for chunk in response.iter_content(chunk_size=8192)
                if output.write(chunk) or True
            ):
                elapsed   = perf_counter() - start
                rate_mbps = (downloaded / elapsed) / 1_000_000 if elapsed > 0 else 0

                self._update_progress(
                    progress   = progress,
                    task       = task,
                    filename   = filename,
                    bytes_done = downloaded,
                    rate_mbps  = rate_mbps
                )

    def _update_progress(
        self,
        progress    : Progress,
        task        : TaskID,
        filename    : str,
        bytes_done  : int,
        rate_mbps   : float = 0,
        status      : str = ""
    ):
        """
        Update progress bar with consistent formatting.

        Args:
            progress   : Progress context manager
            task       : Task ID from progress.add_task
            filename   : Display name for the file
            bytes_done : Bytes completed so far
            rate_mbps  : Transfer rate in MB/s
            status     : Optional status message
        """
        progress.update(
            completed   = bytes_done,
            description = (
                f"Downloading {filename}"
                f"{f' - {status}' if status else ''}"
                f"{f' - {rate_mbps:.1f} MB/s' if rate_mbps > 0 else ''}"
            ),
            task_id     = task
        )

    def run(
        self,
        sample    : bool = False,
        wrf_sfire : bool = False
    ):
        """
        Executes the download workflow.

        Shows source selection or proceeds with specified source.

        Args:
            sample    : If True, download sample dataset
            wrf_sfire : If True, browse full FRDR dataset
        """
        self.ui.print_header("Data Acquisition")

        source = (
            "sample"    if sample else
            "wrf-sfire" if wrf_sfire else
            self.cfg.download.source
        )

        if not source:
            source = self.prompts.select_from_list(
                choices = [
                    (
                        "sample",
                        "Sample Dataset\n  → 468 MB • Single file for quick testing"
                    ),
                    (
                        "wrf-sfire",
                        "Full FRDR Dataset\n  → 5.3 TB • 147 wildfire simulation files"
                    )
                ],
                message = "What would you like to download?"
            )

        if source == "sample":
            self._download_sample()
            return
        try:
            globus_client   = self._ensure_authentication()
            available_files = self._get_available_files(globus_client)

        except Exception as e:
            self.ui.print_message(f"Unable to connect to Globus: {str(e)}", "error")
            return

        file_status = self._get_download_status(available_files)

        self.ui.print_section("FRDR Dataset Browser", minor=True)
        self.ui.print_message(
            message  = "Moisseeva (2020) LES Wildfire Plume Dataset",
            msg_type = "info"
        )
        self.ui.display_download_summary(available_files, file_status)

        selected_file = self.prompts.select_file_from_pages(
            available_files = available_files,
            file_status     = file_status,
            title_prefix    = "Available Files"
        )

        if selected_file:
            self._handle_file_selection(
                file_info     = selected_file,
                file_status   = file_status,
                globus_client = globus_client
            )
        else:
            self.ui.print_message("Download cancelled", "warning")
