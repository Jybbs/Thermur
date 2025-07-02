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
        self.cfg = cfg
        if cfg:
            self.messages          = cfg.messages
            self.wandb_integration = cfg.wandb_integration

    def _safe_import(
        self,
        attr    : str,
        package : str
    ) -> str | None:
        """
        Get a package attribute, handling import errors gracefully.

        Args:
            attr    : The attribute to retrieve from the package.
            package : The name of the package to import.

        Returns:
            The attribute's value, or None if import/access fails.
        """
        try:
            return getattr(__import__(package), attr)
        
        except (ImportError, AttributeError):
            return None

    def _safe_version(
        self,
        package  : str,
        fallback : str = None
    ) -> str | None:
        """
        Get a package version, handling errors gracefully.

        Args:
            package  : The name of the package to check.
            fallback : The value to return if the package is not found.

        Returns:
            The package version string, or the fallback value.
        """
        try:
            return version(package)
        
        except PackageNotFoundError:
            return fallback
        
    def check_wandb_status(self, cfg: DictConfig = None) -> tuple[str, str]:
        """
        Check wandb installation and login status.

        Args:
            cfg: Full configuration object. If None, uses the instance's cfg.

        Returns:
            A tuple of (status, details) for wandb integration.
        """
        if cfg is None:
            cfg = self.cfg
            
        info      = self.get_system_info(cfg.wandb_integration)
        wandb_cfg = cfg.wandb_integration

        if not info["wandb_installed"]:
            return (
                "[red]❌ Not Installed[/red]",
                "[yellow]Run 'poetry install'[/yellow]",
            )

        if info["wandb_user"]:
            return (
                "[green]✅ Connected[/green]",
                f"[cyan]@{info['wandb_user']}[/cyan]",
            )

        api_key_exists = os.environ.get(wandb_cfg.api_key_env) or api.api_key
        if api_key_exists:
            return (
                "[green]✅ API Key Set[/green]",
                "[white]Ready to track[/white]",
            )

        return (
            "[yellow]⚠️  Not Connected[/yellow]",
            "[yellow]Run 'wandb login'[/yellow]",
        )

    def get_system_info(
        self,
        wandb_integration : DictConfig | None = None
    ) -> dict[str, str | int | float | bool | None]:
        """
        Gather comprehensive system information using platform tools.

        Args:
            wandb_integration: Wandb-related configuration. If None, uses
                               the instance's wandb_integration.

        Returns:
            A dictionary containing system details.
        """
        if wandb_integration is None:
            wandb_integration = self.wandb_integration
            
        cuda_available = cuda.is_available()
        info = {
            "cuda"                : cuda_available,
            "device_count"        : cuda.device_count() if cuda_available else 0,
            "mujoco"              : self._safe_import(
                attr="__version__", package="mujoco"
            ),
            "platform"            : platform(),
            "python"              : python_version(),
            "python_version_info" : version_info,
            "thermur"             : self._safe_version(
                package="thermur", fallback="dev"
            ),
            "torch"               : torch_version,
        }

        if cuda_available:
            props                = cuda.get_device_properties(0)
            info["cuda_version"] = cuda.version.cuda
            info["gpu_memory"]   = f"{props.total_memory / 1e9:.1f}GB"
            info["gpu_name"]     = cuda.get_device_name(0)

        try:
            from psutil import virtual_memory
            mem                      = virtual_memory()
            info["memory_available"] = mem.available / 1e9
            info["memory_total"]     = mem.total     / 1e9
        except ImportError:
            info["memory_available"] = 0
            info["memory_total"]     = 0

        try:
            usage                  = disk_usage(".")
            info["disk_available"] = usage.free  / 1e9
            info["disk_total"]     = usage.total / 1e9
        except Exception:
            info["disk_available"] = 0
            info["disk_total"]     = 0

        info["wandb_installed"] = self._safe_import(
            attr="__version__", package="wandb"
        ) is not None
        info["wandb_user"]      = None
        api_key_exists          = False
        if info["wandb_installed"]:
            api_key_exists = os.environ.get(wandb_integration.api_key_env) or api.api_key

        if api_key_exists:
            try:
                user = Api().viewer
                info["wandb_user"] = user.get("username") if user else None
            except Exception:
                info["wandb_user"] = None

        return info

    def get_wandb_url(
        self,
        project           : str,
        wandb_integration : DictConfig | None = None
    ) -> str | None:
        """
        Generate wandb project URL if possible.

        Args:
            project           : The name of the wandb project.

            wandb_integration : Wandb configuration from DictConfig.

        Returns:
            The URL to the wandb project dashboard, or None if not available.
        """
        if wandb_integration is None:
            wandb_integration = self.wandb_integration
            
        info = self.get_system_info(wandb_integration)

        if not info["wandb_installed"]:
            return None

        if info["wandb_user"]:
            return f"https://wandb.ai/{info['wandb_user']}/{project}"
        
        return None

    def validate_config_overrides(
        self,
        overrides         : list[str] | None,
        messages          : DictConfig | None = None,
        wandb_integration : DictConfig | None = None
    ) -> list[str]:
        """
        Validate Hydra configuration override syntax.

        Args:
            overrides         : A list of configuration overrides to validate.
            messages          : Messages configuration. If None, uses instance's messages.
            wandb_integration : Wandb configuration. If None, uses instance's wandb_integration.

        Returns:
            A list of validation issues found; empty if all are valid.
        """
        if not overrides:
            return []

        issues = []
        for o in overrides:
            if "=" not in o:
                issues.append(f"{messages.validation['invalid_override_format']}: {o}")
                continue

            key            = o.split("=")[0]
            sanitized_key  = key.lstrip("+").replace(".", "").replace("_", "")
            key_is_invalid = not sanitized_key.isalnum()
            if key_is_invalid:
                issues.append(f"{messages.validation['invalid_override_key']}: {o}")

        if messages is None:
            messages = self.messages
        if wandb_integration is None:
            wandb_integration = self.wandb_integration
            
        if not self.get_system_info(wandb_integration)["cuda"]:
            issues.append(messages.validation['gpu_unavailable'])

        return issues