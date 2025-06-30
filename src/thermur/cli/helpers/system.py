"""
System utilities for the Thermur CLI application.

This module provides functions for gathering system diagnostics, including
hardware, software, and package information. It is responsible for collecting
the raw data that other modules, like the UI, will then format and display.
"""
import os

from functools          import lru_cache
from importlib.metadata import PackageNotFoundError, version
from omegaconf          import DictConfig
from platform           import platform, python_version
from shutil             import disk_usage
from sys                import version_info
from torch              import cuda, __version__ as torch_version
from wandb              import Api, api


class SystemInspector:
    """
    Provides static methods for system diagnostics and resource validation.

    This class acts as a stateless utility for querying the host environment.
    Its methods are static because they do not depend on any instance-specific
    state. For methods that require access to configuration values (e.g., for
    validation rules or API keys), DictConfig objects are passed in as
    parameters, adhering to a dependency injection pattern without requiring
    an instance of the class.
    """
    @staticmethod
    def _safe_import(
        attr    : str,
        package : str,
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

    @staticmethod
    def _safe_version(
        package  : str,
        fallback : str = None,
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

    @staticmethod
    @lru_cache(maxsize=1)
    def get_system_info(wandb_config: DictConfig) -> dict[str, any]:
        """
        Gather comprehensive system information using platform tools.

        Args:
            wandb_config: Wandb-related configuration from DictConfig.

        Returns:
            A dictionary containing system details.
        """
        cuda_available = cuda.is_available()
        info = {
            "cuda"                : cuda_available,
            "device_count"        : cuda.device_count() if cuda_available else 0,
            "mujoco"              : SystemInspector._safe_import(attr="__version__", package="mujoco"),
            "platform"            : platform(),
            "python"              : python_version(),
            "python_version_info" : version_info,
            "thermur"             : SystemInspector._safe_version(package="thermur", fallback="dev"),
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
            usage              = disk_usage(".")
            info["disk_free"]  = usage.free  / 1e9
            info["disk_total"] = usage.total / 1e9
        except Exception:
            info["disk_free"]  = 0
            info["disk_total"] = 0

        info["wandb_installed"] = SystemInspector._safe_import(attr="__version__", package="wandb") is not None
        info["wandb_user"]      = None
        api_key_exists          = False
        if info["wandb_installed"]:
            api_key_exists = os.environ.get(wandb_config.api_key_env) or api.api_key

        if api_key_exists:
            old_mode = os.environ.get(wandb_config.mode_env)
            os.environ[wandb_config.mode_env] = "offline"

            try:
                user               = Api().viewer
                info["wandb_user"] = user.get("username") if user else None
            except Exception:
                info["wandb_user"] = None  
            finally:
                if old_mode:
                    os.environ[wandb_config.mode_env] = old_mode
                else:
                    os.environ.pop(wandb_config.mode_env, None)

        return info

    @staticmethod
    def check_wandb_status(messages: DictConfig) -> tuple[str, str]:
        """
        Check wandb installation and login status.

        Args:
            messages: Messages configuration containing wandb status messages.

        Returns:
            A tuple of (status, details) for wandb integration.
        """
        info = SystemInspector.get_system_info(messages.wandb_display)

        if not info["wandb_installed"]:
            return (
                messages.wandb_status_not_installed,
                messages.wandb_details_not_installed,
            )

        if info["wandb_user"]:
            return (
                messages.wandb_status_connected,
                messages.wandb_details_connected.format(user=info["wandb_user"]),
            )

        api_key_exists = os.environ.get(messages.wandb_display.api_key_env) or api.api_key
        if api_key_exists:
            return (
                messages.wandb_status_api_key,
                messages.wandb_details_api_key,
            )

        return (
            messages.wandb_status_not_connected,
            messages.wandb_details_not_connected,
        )

    @staticmethod
    def get_wandb_url(wandb_config: DictConfig, ui_config: DictConfig, project: str = "thermur") -> str | None:
        """
        Generate wandb project URL if possible.

        Args:
            wandb_config : Wandb configuration from DictConfig.
            ui_config    : UI configuration from DictConfig.
            project      : The name of the wandb project.

        Returns:
            The URL to the wandb project dashboard, or None if not available.
        """
        info = SystemInspector.get_system_info(wandb_config)

        if not info["wandb_installed"]:
            return None

        if info["wandb_user"]:
            return f"https://wandb.ai/{info['wandb_user']}/{project}"

        entity = os.environ.get(wandb_config.entity_env)
        return (
            f"https://wandb.ai/{entity}/{project}"
            if entity
            else f"https://wandb.ai/{ui_config.wandb_url_placeholder}/{project}"
        )

    @staticmethod
    def validate_config_overrides(overrides: list[str] | None, system_config: DictConfig, wandb_config: DictConfig) -> list[str]:
        """
        Validate Hydra configuration override syntax.

        Args:
            overrides     : A list of configuration overrides to validate.
            system_config : System configuration from DictConfig.
            wandb_config  : Wandb configuration from DictConfig.

        Returns:
            A list of validation issues found; empty if all are valid.
        """
        if not overrides:
            return []

        issues = []
        for o in overrides:
            if "=" not in o:
                issues.append(f"{system_config.invalid_override_format}: {o}")
                continue

            key            = o.split("=")[0]
            sanitized_key  = key.lstrip("+").replace(".", "").replace("_", "")
            key_is_invalid = not sanitized_key.isalnum()
            if key_is_invalid:
                issues.append(f"{system_config.invalid_override_key}: {o}")

        if not SystemInspector.get_system_info(wandb_config)["cuda"]:
            issues.append(system_config.gpu_unavailable)

        return issues