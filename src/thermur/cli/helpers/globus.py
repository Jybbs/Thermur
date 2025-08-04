"""
Globus authentication and data transfer orchestration for large-scale datasets.

This module provides a high-level interface to the Globus SDK, managing OAuth2
authentication flows, endpoint operations, and asynchronous transfer tasks. It
abstracts the complexity of Globus operations while maintaining the flexibility
needed for scientific data workflows.

The manager handles:
- Browser-based OAuth2 authentication with automatic token refresh
- File listing and metadata retrieval from remote endpoints
- Transfer task submission between Globus endpoints
- Progress monitoring for long-running transfers
"""
from __future__   import annotations
from config.cli   import GlobusSecrets
from config.types import EndpointInfo, FileInfo, TransferStatus
from contextlib   import suppress
from globus_sdk   import NativeAppAuthClient, RefreshTokenAuthorizer, TransferClient, TransferData
from pathlib      import Path
from time         import perf_counter, sleep
from typing       import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from config.cli import DownloadModel


class GlobusManager:
    """
    High-level orchestrator for Globus authentication and data operations.

    This class provides a simplified interface to Globus functionality while
    handling the complexity of OAuth2 flows, token management, and transfer
    operations. It maintains authentication state across CLI invocations
    through secure token persistence.

    The manager follows a lazy authentication pattern - tokens are loaded
    from storage when available, and users are only prompted to authenticate
    when necessary. This provides a seamless experience for repeated use.
    """

    def __init__(self, download: DownloadModel):
        """
        Initialize the Globus manager with download configuration.

        Sets up the OAuth2 client and loads any existing authentication secrets.
        Authentication is attempted lazily when needed. All token management is
        handled internally by this class.

        Args:
            download: Download configuration containing Globus settings
        """
        self.client_id    = download.globus_client_id
        self.dataset_path = download.globus_dataset_path
        self.endpoint_id  = download.globus_endpoint_id
        self.scopes       = download.globus_scopes
        self.secrets      = GlobusSecrets()

    def _save_secrets(self):
        """
        Save authentication secrets to the filesystem.

        Creates individual files for each token field in the secrets directory.
        This allows Pydantic's secrets_dir functionality to automatically load
        these values on the next instantiation.
        """
        Path(self.secrets.secrets_path).mkdir(exist_ok=True, parents=True)

        for field_name, value in self.secrets.model_dump(
            exclude      = {'is_valid', 'secrets_path'},
            exclude_none = True,
            mode         = 'python'
        ).items():
            if hasattr(value, 'get_secret_value'):
                value = value.get_secret_value()

            if value is not None:
                file_path = Path(self.secrets.secrets_path) / field_name
                file_path.write_text(str(value))
                with suppress(OSError):
                    file_path.chmod(0o600)

    def _monitor_transfer_task(
        self,
        task_id         : str,
        transfer_client : TransferClient
    ) -> TransferStatus:
        """
        Check the current status of a transfer task.

        Queries the Globus service for detailed information about a submitted
        transfer task. This includes completion status, bytes transferred,
        and any error information.

        Args:
            task_id         : UUID of the transfer task
            transfer_client : Authenticated client for API calls

        Returns:
            Task status dictionary containing:
            - bytes_transferred : Number of bytes successfully transferred
            - files_transferred : Number of files completed
            - is_ok             : Boolean indicating if transfer completed
            - mbps              : Current transfer rate in MB/s
            - nice_status       : Human-readable status message
            - status            : Current status (ACTIVE, SUCCEEDED, FAILED)
        """
        task          = transfer_client.get_task(task_id)
        bytes_per_sec = task.get("effective_bytes_per_second", 0)

        return TransferStatus(
            bytes_transferred = task.get("bytes_transferred", 0),
            files_transferred = task.get("files_transferred", 0),
            is_ok             = task.get("is_ok", False),
            mbps              = bytes_per_sec / (1024 * 1024),
            nice_status       = task.get("nice_status", "Unknown"),
            status            = task["status"]
        )

    def finalize_oauth2_flow(
        self,
        auth_code : str,
        client    : NativeAppAuthClient
    ) -> TransferClient:
        """
        Complete OAuth2 authentication with the provided code.

        Args:
            auth_code : Authorization code from the browser
            client    : The NativeAppAuthClient from start_oauth2_flow

        Returns:
            Authenticated TransferClient with fresh tokens
        """
        token_response  = client.oauth2_exchange_code_for_tokens(auth_code)
        transfer_tokens = token_response.by_resource_server["transfer.api.globus.org"]
        self.secrets    = GlobusSecrets(
            refresh_token = transfer_tokens.get("refresh_token"),
            scope         = transfer_tokens.get("scope")
        )

        self._save_secrets()

        return TransferClient(
            authorizer = RefreshTokenAuthorizer(
                auth_client   = client,
                refresh_token = transfer_tokens["refresh_token"]
            )
        )

    def get_local_endpoints(self, transfer_client: TransferClient) -> list[EndpointInfo]:
        """
        Retrieve all Globus Connect Personal endpoints for the authenticated user.

        Queries the Globus service for endpoints where the current user has
        ownership. This is typically used to find the local endpoint created
        by Globus Connect Personal for receiving transferred files.

        Args:
            transfer_client : Authenticated client for API calls

        Returns:
            List of endpoint information dictionaries containing:
            - id           : Endpoint UUID
            - display_name : Human-readable endpoint name
            - description  : Optional endpoint description
        """
        return [
            EndpointInfo(
                display_name = ep["display_name"],
                id           = ep["id"]
            )
            for ep in transfer_client.endpoint_search(
                filter_scope = "my-endpoints"
            )
        ]

    def get_or_create_client(self) -> TransferClient | None:
        """
        Obtain an authenticated Transfer client, handling all OAuth2 flows.

        This method implements lazy authentication with automatic token refresh.
        If valid secrets exist in the application context, they are used to create
        an authorizer that will automatically refresh expired tokens. If no secrets
        exist or refresh fails, the user is guided through a browser-based OAuth2 flow.

        The authentication flow:
        1. Check for existing secrets in application context
        2. If found and valid, create authorizer with refresh capability
        3. If missing or invalid, perform browser-based OAuth2
        4. Update application context with new secrets

        Returns:
            Authenticated TransferClient ready for operations

        Raises:
            Exception: If authentication fails after user interaction
        """
        if self.secrets and self.secrets.is_valid and self.secrets.refresh_token:
            auth_client = NativeAppAuthClient(self.client_id)
            authorizer  = RefreshTokenAuthorizer(
                auth_client   = auth_client,
                refresh_token = self.secrets.refresh_token.get_secret_value()
            )

            return TransferClient(authorizer=authorizer)

        # If no valid secrets, auth needs to be handled by caller
        return None

    def list_endpoint_directory(
        self,
        endpoint_id     : str,
        path            : str,
        transfer_client : TransferClient
    ) -> list[FileInfo]:
        """
        List files and directories at a specific path on a Globus endpoint.

        Performs a directory listing operation on the specified endpoint,
        returning metadata for all items at the given path. This is used
        to browse available datasets and verify transfer destinations.

        Args:
            endpoint_id     : UUID of the Globus endpoint
            path            : Absolute path within the endpoint
            transfer_client : Authenticated client for API calls

        Returns:
            List of item dictionaries containing:
            - name: File or directory name
            - path: Full path to the item
            - size: Size in bytes (files only)
            - type: Either "file" or "dir"

        Raises:
            globus_sdk.GlobusAPIError: If endpoint or path is inaccessible
        """
        response = transfer_client.operation_ls(endpoint_id, path=path)

        return [
            FileInfo(
                name = item["name"],
                path = str(Path(path) / item['name']),
                size = item.get("size", 0),
                type = item["type"]
            )
            for item in response
        ]

    def submit_transfer_task(
        self,
        dest_endpoint   : str,
        items           : list[tuple[str, str]],
        label           : str,
        source_endpoint : str,
        transfer_client : TransferClient
    ) -> str:
        """
        Submit an asynchronous transfer task to the Globus service.

        Creates and submits a transfer task that will be executed by the
        Globus infrastructure. The transfer runs asynchronously, allowing
        for reliable transfer of large datasets without maintaining a
        connection.

        Args:
            dest_endpoint   : UUID of destination endpoint
            items           : List of (source_path, dest_path) tuples to transfer
            label           : Human-readable label for the transfer task
            source_endpoint : UUID of source endpoint
            transfer_client : Authenticated client for API calls

        Returns:
            Task UUID for monitoring transfer progress

        Raises:
            globus_sdk.GlobusAPIError: If submission fails
        """
        transfer_data = TransferData(
            transfer_client,
            destination_endpoint = dest_endpoint,
            label                = label,
            source_endpoint      = source_endpoint,
            sync_level           = 1
        )

        for source_path, dest_path in items:
            transfer_data.add_item(source_path, dest_path)

        result = transfer_client.submit_transfer(transfer_data)
        return result["task_id"]

    def start_oauth2_flow(self) -> tuple[NativeAppAuthClient, str]:
        """
        Start OAuth2 flow and return the authentication URL.

        Initiates the OAuth2 native app flow and generates the authorization
        URL for browser-based authentication. The returned client object must
        be preserved to complete the flow after user authorization.

        Returns:
            Tuple of (auth_client, auth_url) for the OAuth2 flow
        """
        client = NativeAppAuthClient(self.client_id)
        client.oauth2_start_flow(
            requested_scopes=self.scopes,
            refresh_tokens=True  # Enable refresh tokens
        )
        return client, client.oauth2_get_authorize_url()

    def wait_for_transfer(
        self,
        task_id           : str,
        transfer_client   : TransferClient,
        polling_interval  : int  = 10,
        progress_callback : Callable[[TransferStatus], None] | None = None,
        timeout           : int | None = None
    ) -> bool:
        """
        Block until a transfer task completes or times out.

        Polls the Globus service at regular intervals to check task status.
        This is useful for smaller transfers where synchronous behavior is
        desired. For large transfers, async monitoring is recommended.

        Args:
            task_id           : UUID of the transfer task to monitor
            transfer_client   : Authenticated client for API calls
            polling_interval  : Seconds between status checks
            progress_callback : Optional callback function that receives status dict
            timeout           : Maximum seconds to wait (None for infinite)

        Returns:
            True if transfer succeeded, False if failed or timed out
        """
        start_time = perf_counter()

        while status := self._monitor_transfer_task(task_id, transfer_client):
            if progress_callback:
                progress_callback(status)

            match status["status"]:
                case "SUCCEEDED" : return True
                case "FAILED"    : return False
                case _           : pass  # Continue polling for ACTIVE or other states

            if timeout and (perf_counter() - start_time) > timeout:
                return False

            sleep(polling_interval)

        # If we exit the while loop without returning, the transfer failed
        return False
