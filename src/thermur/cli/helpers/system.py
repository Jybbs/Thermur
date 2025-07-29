"""
System utilities for the Thermur CLI application.

This module provides functions for gathering system diagnostics, including
hardware, software, and package information. It is responsible for collecting
the raw data that other modules, like the UI, will then format and display.
"""
from contextlib         import suppress
from importlib.metadata import PackageNotFoundError, version
from omegaconf          import DictConfig
from pathlib            import Path
from platform           import platform, python_version
from shutil             import disk_usage
from sys                import version_info
from torch              import __version__ as torch_version, cuda

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
        self.download = cfg.download

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

    def _get_dataset_info(self) -> dict[str, any]:
        """
        Gather information about downloaded dataset files.
        
        Returns:
            Dictionary with dataset_size in GB, dataset_count, and has_sample.
        """
        with suppress(Exception):
            all_files   = self._get_wrf_files()
            sample_path = self.download.sample_data_path
            
            if sample_path.exists():
                all_files.append(sample_path)
            
            return {
                "dataset_count" : len(all_files),
                "dataset_size"  : sum(f.stat().st_size for f in all_files) / 1e9,
                "has_sample"    : sample_path.exists(),
            }
        return {"dataset_count": 0, "dataset_size": 0.0, "has_sample": False}
    
    def _get_disk_info(self) -> dict[str, float]:
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

    def _get_memory_info(self) -> dict[str, float]:
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
        with suppress(PackageNotFoundError):
            return version(package_name)
        return default

    def _get_wrf_files(self) -> list[Path]:
        """
        Get list of WRF-SFIRE NetCDF files from configured directory.
        
        Returns:
            List of Path objects for WRF files, empty list if none found.
        """
        wrf_dir = self.download.wrf_sfire_dir
        if not wrf_dir.exists():
            return []
            
        return [f for f in wrf_dir.glob("*.nc") if f.is_file()]


    def get_system_info(self) -> dict[str, any]:
        """
        Gather comprehensive system information.

        Collects information about installed packages, hardware capabilities,
        and system resources. This includes Python version, PyTorch details,
        CUDA availability, memory, and disk usage.

        Returns:
            Dictionary containing all system information with keys:
            - Package versions : mujoco, thermur, torch
            - System info      : platform, python, python_version_info
            - Hardware         : cuda info, memory stats, disk usage
        """
        info = {
            "mujoco"              : self._get_package_version("mujoco"),
            "platform"            : platform(),
            "python"              : python_version(),
            "python_version_info" : version_info,
            "thermur"             : self._get_package_version("thermur", "dev"),
            "torch"               : torch_version,
        }
        
        info.update(self._get_cuda_info())
        info.update(self._get_memory_info())
        info.update(self._get_disk_info())
        info.update(self._get_dataset_info())
        
        return info

    
    def resolve_data_path(self, use_sample: bool = False) -> tuple[Path, str]:
        """
        Resolves the appropriate data path based on availability and user preference.
        
        This method implements a fallback strategy for data selection:
        1. If sample explicitly requested and exists          -> use sample
        2. If WRF-SFIRE data exists and not requesting sample -> use first WRF file
        3. If no WRF data but sample exists                   -> fallback to sample  
        4. Otherwise                                          -> no data available
        
        Args:
            use_sample : Whether the user explicitly requested sample data
            
        Returns:
            Tuple of (data_path, status_message)
            
        Raises:
            FileNotFoundError: If no data is available for training
        """
        if not hasattr(self, 'download'):
            raise ValueError("SystemInspector missing download configuration")
            
        sample_path = self.download.sample_data_path
        wrf_files   = [] if use_sample else self._get_wrf_files()
        
        match (use_sample, bool(wrf_files), sample_path.exists()):
            case (True, _, True):
                return sample_path, "Using sample dataset as requested."
            case (False, True, _):
                return wrf_files[0], f"Using WRF-SFIRE data: {wrf_files[0].name}"
            case (False, False, True):
                return sample_path, "No WRF-SFIRE data found. Using sample data."
            case _:
                raise FileNotFoundError(
                    "No training data available. "
                    "Run 'thermur download' to get sample data."
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

        issues = []
        
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
        
        if not cuda.is_available():
            issues.append("GPU not available - training will be slower on CPU")

        return issues