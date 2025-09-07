"""
System utilities for the Thermur CLI application.

This module provides functions for gathering system diagnostics, including
hardware, software, and package information. It is responsible for collecting
the raw data that other modules, like the UI, will then format and display.
"""
from __future__         import annotations
from contextlib         import suppress
from importlib.metadata import PackageNotFoundError, version
from omegaconf          import OmegaConf
from pathlib            import Path
from platform           import platform, python_version
from shutil             import disk_usage
from sys                import version_info
from typing             import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from config.cli.builds import CLIConfiguration
    from config.types      import SystemInfo


class SystemInspector:
    """
    Provides methods for system diagnostics and resource validation.

    This class manages system inspection with access to configuration,
    reducing the need to pass configuration objects to every method call.
    """

    def __init__(self, cfg: CLIConfiguration):
        """
        Initialize the system inspector with configuration.

        Args:
            cfg: The CLI configuration object containing all settings.
        """
        self._torch_cache: dict[str, Any] | None = None

    def _get_cuda_info(self) -> SystemInfo:
        """
        Gather CUDA-related system information.

        Returns:
            Dictionary containing CUDA availability, device count, GPU name,
            memory, and CUDA version if available.
        """
        torch_info = self._get_torch()
        if not torch_info["available"]:
            return {"cuda": False, "device_count": 0}
            
        cuda = torch_info["cuda"]
        if not cuda.is_available():
            return {"cuda": False, "device_count": 0}

        torch_version = torch_info["version"]
        return {
            "cuda"         : True,
            "cuda_version" : (
                torch_version.split('+')[0] if '+' in torch_version else torch_version
            ),
            "device_count" : cuda.device_count(),
            "gpu_memory"   : f"{cuda.mem_get_info(0)[1] / 1e9:.1f}GB",
            "gpu_name"     : cuda.get_device_name(0),
        }

    def _get_dataset_info(self) -> SystemInfo:
        """
        Gather information about available NetCDF dataset files.

        Returns:
            Dictionary with dataset_size in GB and dataset_count.
        """
        from thermur.imitation.controller import DemonstrationsDataset
        
        relative_paths = DemonstrationsDataset._find_netcdf_files()
        
        if relative_paths:
            total_size = sum(
                (Path("data/raw") / p).stat().st_size 
                for p in relative_paths
            ) / 1e9
            
            return {
                "dataset_count" : len(relative_paths),
                "dataset_size"  : total_size,
            }
        
        return {"dataset_count": 0, "dataset_size": 0.0}

    def _get_disk_info(self) -> SystemInfo:
        """
        Gather disk usage information for the current directory.

        Returns:
            Dictionary with disk_available and disk_total in GB.
            Returns zeros if disk information cannot be retrieved.
        """
        with suppress(Exception):
            usage = disk_usage(".")
            return {
                "disk_available" : usage.free  / 1e9,
                "disk_total"     : usage.total / 1e9,
            }
        return {"disk_available": 0, "disk_total": 0}

    def _get_memory_info(self) -> SystemInfo:
        """
        Gather system memory information using psutil.

        Returns:
            Dictionary with memory_available and memory_total in GB.
            Returns zeros if psutil is not installed.
        """
        with suppress(ImportError):
            from psutil import virtual_memory
            mem = virtual_memory()
            return {
                "memory_available" : mem.available / 1e9,
                "memory_total"     : mem.total     / 1e9,
            }
        return {"memory_available": 0, "memory_total": 0}

    def _get_package_version(
        self,
        package_name : str,
        default      : str | None = None
    ) -> str | None:
        """
        Get version of an installed package.

        Args:
            package_name : Name of the package to check.
            default      : Default value if package is not found.

        Returns:
            Version string or default value.
        """
        with suppress(PackageNotFoundError):
            return version(package_name)
        return default

    def _get_torch(self) -> dict[str, Any]:
        """
        Lazy import and cache PyTorch modules.

        Returns:
            Dictionary containing PyTorch availability, version, and cuda module.
            Keys: available, version, cuda.
        """
        if self._torch_cache is None:
            try:
                import torch
                self._torch_cache = {
                    "available" : True,
                    "cuda"      : torch.cuda,
                    "version"   : torch.__version__,
                }
            except ImportError:
                self._torch_cache = {
                    "available" : False,
                    "cuda"      : None,
                    "version"   : "not installed",
                }
        return self._torch_cache

    def extract_training_cfg(
        self,
        cfg       : Any,
        overrides : list[str] | None = None
    ) -> dict[str, Any]:
        """
        Extract the main training configuration sections.
        
        Filters the full configuration to only include the three main
        sections used for training: controller, environment, and training.
        This provides a consistent way to prepare configs for display
        or logging across different commands.
        
        Args:
            cfg       : Full configuration object (OmegaConf or dict)
            overrides : Optional list of override strings
            
        Returns:
            Dictionary with extracted config sections and overrides
        """
        if not isinstance(
            container := (
                OmegaConf.to_container(cfg, resolve=True) 
                if hasattr(cfg, '_metadata')
                else cfg
            ), dict
        ):
            container = {}
        
        return {
            "controller"  : container.get("controller",  {}),
            "environment" : container.get("environment", {}),
            "training"    : container.get("training",    {}),
            "overrides"   : overrides or []
        }
    
    def get_system_info(self) -> SystemInfo:
        """
        Gather comprehensive system information.

        Collects information about installed packages, hardware capabilities,
        and system resources. This includes Python version, PyTorch details,
        CUDA availability, memory, and disk usage.

        Returns:
            Dictionary containing all system information with keys:
            - Package versions : thermur, torch
            - System info      : platform, python, python_version_info
            - Hardware         : cuda info, memory stats, disk usage
        """
        torch_info = self._get_torch()
            
        base_info: SystemInfo = {
            "platform"            : platform(),
            "python"              : python_version(),
            "python_version_info" : version_info,
            "thermur"             : self._get_package_version("thermur", "dev"),
            "torch"               : torch_info["version"],
        }

        return (
            base_info
            | self._get_cuda_info()
            | self._get_memory_info()
            | self._get_disk_info()
            | self._get_dataset_info()
        )

    def validate_overrides(self, overrides: list[str] | None) -> list[str]:
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

        issues: list[str] = []

        for o in overrides:
            if "=" not in o:
                issues.append(
                    f"Invalid override format (expected key=value): {o}"
                )
                continue

            key = (
                o.partition("=")[0]
                .removeprefix("+").replace(".", "").replace("_", "")
            )
            if not key.isalnum():
                issues.append(
                    f"Invalid override key (must be alphanumeric): {o}"
                )

        torch_info = self._get_torch()
        if not torch_info["available"]:
            issues.append("PyTorch not installed - training requires PyTorch")
        elif not torch_info["cuda"].is_available():
            issues.append("GPU not available - training will be slower on CPU")

        return issues
