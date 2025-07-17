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
from configs.cli import GlobusSecrets
from globus_sdk  import NativeAppAuthClient, RefreshTokenAuthorizer, TransferClient, TransferData
from omegaconf   import DictConfig
from typing      import Optional

import time
import webbrowser


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
    
    def __init__(self, download: DictConfig):
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

    def _do_native_app_authentication(self) -> TransferClient:
        """
        Perform interactive OAuth2 authentication via browser.
        
        Implements the OAuth2 native app flow by opening the user's browser
        to the Globus authentication page. After the user approves access,
        they paste the authorization code back into the CLI to complete
        the flow.
        
        Returns:
            Authenticated TransferClient with fresh tokens
            
        Raises:
            Exception: If the user cancels or authentication fails
        """
        client = NativeAppAuthClient(self.client_id)
        client.oauth2_start_flow(requested_scopes=self.scopes)
        
        auth_url = client.oauth2_get_authorize_url()
        
        print("\nAuthentication required for Globus access.")
        print("Your browser will open to complete authentication.\n")
        print("If the browser doesn't open automatically, please visit:")
        print(f"  {auth_url}\n")
        
        try:
            webbrowser.open(auth_url, new=2)
        except:
            pass
        
        auth_code       = input("Enter the authorization code from the browser: ")
        token_response  = client.oauth2_exchange_code_for_tokens(auth_code)
        transfer_tokens = token_response.by_resource_server["transfer.api.globus.org"]

        # Create new secrets instance
        self.secrets = GlobusSecrets(
            access_token  = transfer_tokens["access_token"],
            expires_at    = transfer_tokens["expires_at"],
            refresh_token = transfer_tokens["refresh_token"],
            scope         = transfer_tokens["scope"]
        )
        
        # Save to disk for future use
        self.secrets.save()
        
        authorizer = RefreshTokenAuthorizer(
            auth_client   = client,
            refresh_token = self.secrets.refresh_token.get_secret_value()
        )
        
        return TransferClient(authorizer=authorizer)
    
    def authenticate(self) -> TransferClient:
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
        if self.secrets and self.secrets.is_valid:
            auth_client = NativeAppAuthClient(self.client_id)
            authorizer  = RefreshTokenAuthorizer(
                access_token  = self.secrets.access_token.get_secret_value(),
                auth_client   = auth_client,
                expires_at    = self.secrets.expires_at,
                refresh_token = self.secrets.refresh_token.get_secret_value()
            )
            
            # The authorizer will automatically refresh on first use if needed
            return TransferClient(authorizer=authorizer)
        
        return self._do_native_app_authentication()
    
    def get_local_endpoints(self, transfer_client: TransferClient) -> list[dict]:
        """
        Retrieve all Globus Connect Personal endpoints for the authenticated user.
        
        Queries the Globus service for endpoints where the current user has
        ownership. This is typically used to find the local endpoint created
        by Globus Connect Personal for receiving transferred files.
        
        Args:
            transfer_client : Authenticated client for API calls
            
        Returns:
            List of endpoint information dictionaries containing:
            - id: Endpoint UUID
            - display_name: Human-readable endpoint name  
            - description: Optional endpoint description
        """
        # Filter for GCP endpoints owned by the user
        endpoints = transfer_client.endpoint_search(
            filter_scope = "my-endpoints",
            filter_type  = "GCP"
        )
        
        return [
            {
                "description"  : ep.get("description", ""),
                "display_name" : ep["display_name"],
                "id"           : ep["id"]
            }
            for ep in endpoints
        ]
    
    def list_endpoint_directory(
        self,
        endpoint_id     : str,
        path            : str,
        transfer_client : TransferClient
    ) -> list[dict]:
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
            {
                "name" : item["name"],
                "path" : f"{path.rstrip('/')}/{item['name']}",
                "size" : item.get("size", 0),
                "type" : item["type"]
            }
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
            sync_level           = "checksum"  # Verify integrity
        )
        
        for source_path, dest_path in items:
            transfer_data.add_item(source_path, dest_path)
        
        result = transfer_client.submit_transfer(transfer_data)
        return result["task_id"]
    
    def monitor_transfer_task(
        self,
        task_id         : str, 
        transfer_client : TransferClient
    ) -> dict:
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
            - nice_status       : Human-readable status message
            - status            : Current status (ACTIVE, SUCCEEDED, FAILED)
        """
        task = transfer_client.get_task(task_id)
        
        return {
            "bytes_transferred" : task.get("bytes_transferred", 0),
            "files_transferred" : task.get("files_transferred", 0), 
            "is_ok"             : task.get("is_ok", False),
            "nice_status"       : task.get("nice_status", "Unknown"),
            "status"            : task["status"]
        }
    
    def wait_for_transfer(
        self,
        task_id         : str,
        transfer_client : TransferClient,
        polling_interval: int           = 10,
        timeout         : Optional[int] = None
    ) -> bool:
        """
        Block until a transfer task completes or times out.
        
        Polls the Globus service at regular intervals to check task status.
        This is useful for smaller transfers where synchronous behavior is
        desired. For large transfers, async monitoring is recommended.
        
        Args:
            task_id          : UUID of the transfer task to monitor
            transfer_client  : Authenticated client for API calls
            polling_interval : Seconds between status checks
            timeout          : Maximum seconds to wait (None for infinite)
            
        Returns:
            True if transfer succeeded, False if failed or timed out
        """
        start_time = time.time()
        
        while True:
            status = self.monitor_transfer_task(task_id, transfer_client)
            
            if status["status"] == "SUCCEEDED":
                return True
            elif status["status"] == "FAILED":
                return False
                
            if timeout and (time.time() - start_time) > timeout:
                return False
                
            time.sleep(polling_interval)
