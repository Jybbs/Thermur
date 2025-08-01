"""
Type definitions for the Thermur CLI.

This module consolidates all type protocols, TypedDicts, and type aliases used
throughout the CLI, providing a single source of truth for type definitions.
"""
from config.cli.schemas import *
from typing             import Protocol, TypedDict


class CLIConfig(Protocol):
    """
    Protocol defining the structure of CLI configuration.
    
    This represents the SimpleNamespace object containing all CLI-specific
    configuration models. Each attribute is a Pydantic model instance.
    """
    display  : DisplayModel
    download : DownloadModel
    secrets  : GlobusSecrets
    wandb    : WandbConfig


class FileInfo(TypedDict):
    """
    Globus file metadata returned from endpoint directory listings.
    
    Represents file and directory information from Globus Transfer API
    list operations, providing essential metadata for transfer operations.
    """
    name : str  # Filename or directory name
    path : str  # Full path on the endpoint
    size : int  # File size in bytes
    type : str  # Either 'file' or 'dir'


class EndpointInfo(TypedDict):
    """
    Globus endpoint identification information.
    
    Represents a Globus Connect endpoint that can be used as a source
    or destination for data transfers.
    """
    display_name : str  # Human-readable endpoint name
    id           : str  # UUID of the endpoint


class TransferStatus(TypedDict):
    """
    Globus transfer task status information.
    
    Contains detailed status information for monitoring ongoing transfers
    including progress, performance metrics, and completion state.
    """
    bytes_transferred : int   # Number of bytes successfully transferred
    files_transferred : int   # Number of files completed
    is_ok             : bool  # Boolean indicating if transfer completed
    mbps              : float # Current transfer rate in MB/s
    nice_status       : str   # Human-readable status message
    status            : str   # Current status (ACTIVE, SUCCEEDED, FAILED)
