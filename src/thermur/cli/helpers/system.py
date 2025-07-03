"""
System utilities for the Thermur CLI application.

This module provides functions for gathering system diagnostics, including
hardware, software, and package information. It is responsible for collecting
the raw data that other modules, like the UI, will then format and display.
"""
from importlib.metadata import PackageNotFoundError, version
from omegaconf          import DictConfig
from platform           import platform, python_version
from shutil             import disk_usage
from sys                import version_info
from torch              import __version__ as torch_version, cuda
from wandb              import Api, api

import os


class SystemInspector:
    """
    Provides methods for system diagnostics and resource validation.

    This class manages system inspection with access to configuration,
    reducing the need to pass configuration objects to every method call.
    """
    
    def __init__(self, cfg: DictConfig = None):
        """
        Initialize the system inspector with optional configuration.
        
        Args:
            cfg: The configuration object. If provided, commonly used
                 sections are extracted for easier access.
        """
        self.cfg               = cfg
        self.messages          = getattr(cfg, 'messages', None)
        self.wandb_integration = getattr(cfg, 'wandb_integration', None)
        self._wandb_status     = None

    def _get_cuda_info(self) -> dict[str, any]:
        """
        Gather CUDA-related system information.
        
        Returns:
            Dictionary containing CUDA availability, device count, GPU name,
            memory, and CUDA version if available.
        """
        if not cuda.is_available():
            return {"cuda": False, "device_count": 0}
        
        props = cuda.get_device_properties(0)
        return {
            "cuda"         : True,
            "cuda_version" : cuda.version.cuda,
            "device_count" : cuda.device_count(),
            "gpu_memory"   : f"{props.total_memory / 1e9:.1f}GB",
            "gpu_name"     : cuda.get_device_name(0),
        }

    def _get_disk_info(self) -> dict[str, float]:
        """
        Gather disk usage information for the current directory.
        
        Returns:
            Dictionary with disk_available and disk_total in GB.
            Returns zeros if disk information cannot be retrieved.
        """
        try:
            usage = disk_usage(".")
            return {
                "disk_available" : usage.free  / 1e9,
                "disk_total"     : usage.total / 1e9,
            }
        except Exception:
            return {"disk_available": 0, "disk_total": 0}

    def _get_memory_info(self) -> dict[str, float]:
        """
        Gather system memory information using psutil.
        
        Returns:
            Dictionary with memory_available and memory_total in GB.
            Returns zeros if psutil is not installed.
        """
        try:
            from psutil import virtual_memory
            mem = virtual_memory()
            return {
                "memory_available" : mem.available / 1e9,
                "memory_total"     : mem.total     / 1e9,
            }
        except ImportError:
            return {"memory_available": 0, "memory_total": 0}

    def _get_package_version(
        self, 
        package_name : str, 
        default      : str = None
    ) -> str | None:
        """
        Get version of an installed package.
        
        Args:
            package_name : Name of the package to check.
            default      : Default value if package is not found.
            
        Returns:
            Version string or default value.
        """
        try:
            module = __import__(package_name)
            return getattr(module, '__version__', None)
        
        except ImportError:
            try:
                return version(package_name)
            except PackageNotFoundError:
                return default

    def _get_wandb_status(self) -> dict[str, any]:
        """
        Get comprehensive wandb status information.
        
        Checks installation, API key presence, and user authentication
        in a single pass. Results are cached for efficiency.
        
        Returns:
            Dictionary with keys:
            - installed : bool
            - api_key   : str | None
            - username  : str | None
        """
        if self._wandb_status is not None:
            return self._wandb_status
            
        status = {
            "installed" : False,
            "api_key"   : None,
            "username"  : None,
        }
        
        try:
            __import__('wandb')
            status["installed"] = True
        except ImportError:
            self._wandb_status = status
            return status
            
        status["api_key"] = (
            os.environ.get(self.wandb_integration.api_key_env) or 
            api.api_key
        )
        
        if status["api_key"]:
            try:
                user = Api().viewer
                status["username"] = user.get("username") if user else None
            except Exception:
                pass
                
        self._wandb_status = status
        return status

    def check_wandb_status(self) -> tuple[str, str]:
        """
        Check wandb installation and authentication status.

        Returns:
            Tuple of (status_message, details_message) with Rich markup
            indicating the current wandb state and any required actions.
        """
        status = self._get_wandb_status()
        
        if not status["installed"]:
            return (
                "[red]❌ Not Installed[/red]",
                "[yellow]Run 'poetry install'[/yellow]",
            )

        if status["username"]:
            return (
                "[green]✅ Connected[/green]",
                f"[cyan]@{status['username']}[/cyan]",
            )

        if status["api_key"]:
            return (
                "[green]✅ API Key Set[/green]",
                "[white]Ready to track[/white]",
            )

        return (
            "[yellow]⚠️  Not Connected[/yellow]",
            "[yellow]Run 'wandb login'[/yellow]",
        )

    def get_system_info(self) -> dict[str, any]:
        """
        Gather comprehensive system information.

        Collects information about installed packages, hardware capabilities,
        and system resources. This includes Python version, PyTorch details,
        CUDA availability, memory, and disk usage.

        Returns:
            Dictionary containing all system information with keys:
            - Package versions : mujoco, thermur, torch, wandb_installed
            - System info      : platform, python, python_version_info
            - Hardware         : cuda info, memory stats, disk usage
            - wandb_user if authenticated
        """
        wandb_status = self._get_wandb_status()
        
        info = {
            "mujoco"              : self._get_package_version("mujoco"),
            "platform"            : platform(),
            "python"              : python_version(),
            "python_version_info" : version_info,
            "thermur"             : self._get_package_version("thermur", "dev"),
            "torch"               : torch_version,
            "wandb_installed"     : wandb_status["installed"],
            "wandb_user"          : wandb_status["username"],
        }
        
        info.update(self._get_cuda_info())
        info.update(self._get_memory_info())
        info.update(self._get_disk_info())
        
        return info

    def get_wandb_url(self, project: str) -> str | None:
        """
        Generate wandb project URL if user is authenticated.

        Args:
            project: The wandb project name.

        Returns:
            Full URL to the wandb project dashboard if authenticated,
            None otherwise.
        """
        username = self._get_wandb_status()["username"]
        return f"https://wandb.ai/{username}/{project}" if username else None

    def validate_config_overrides(self, overrides: list[str] | None) -> list[str]:
        """
        Validate Hydra configuration override syntax.

        Checks that each override follows the pattern key=value and that
        keys contain only valid characters. Also validates system requirements
        like GPU availability.

        Args:
            overrides: List of Hydra overrides in key=value format.

        Returns:
            List of validation error messages. Empty list indicates all
            overrides are valid.
        """
        if not overrides:
            return []

        issues = []
        
        for o in overrides:
            if "=" not in o:
                issues.append(
                    f"{self.messages.validation['invalid_override_format']}: {o}"
                )
                continue

            key = o.split("=")[0].lstrip("+").replace(".", "").replace("_", "")
            if not key.isalnum():
                issues.append(
                    f"{self.messages.validation['invalid_override_key']}: {o}"
                )
        
        if not cuda.is_available():
            issues.append(self.messages.validation['gpu_unavailable'])

        return issues