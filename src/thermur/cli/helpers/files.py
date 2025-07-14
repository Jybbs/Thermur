"""
File I/O utilities for the Thermur CLI.

This module provides reusable file operations including JSON manifest
management, file size formatting, download operations, and other common
file-related tasks.
"""
from datetime import datetime
from pathlib  import Path
from typing   import Any

import json
import requests


class FileIO:
    """
    Manages file I/O operations for the CLI.
    
    Provides utilities for JSON manifest operations, file downloads,
    size formatting, and other common file-related tasks with proper
    error handling.
    """
    
    def __init__(
        self, 
        cache_dir   : Path, 
        chunk_size  : int = 8192,
        dataset_url : str = None
    ):
        """
        Initialize the file I/O helper.
        
        Args:
            cache_dir   : Directory for caching downloaded files
            chunk_size  : Download chunk size in bytes
            dataset_url : Full URL to the dataset page
        """
        self.cache_dir     = cache_dir
        self.chunk_size    = chunk_size
        self.dataset_url   = dataset_url
        self.manifest_path = cache_dir / "manifest.json"
    
    def check_existing_files(self) -> set[str]:
        """
        Check manifest and cache directory for already downloaded files.
        
        Returns:
            Set of filenames that have already been downloaded
        """
        manifest_files = {f["name"] for f in self.load_json_manifest(self.manifest_path).get("files", [])}
        disk_files = {f.name for f in self.cache_dir.glob("*.nc")} if self.cache_dir.exists() else set()
        return manifest_files | disk_files
    
    def check_file_status(self, file_info: dict) -> tuple[str, int]:
        """
        Check the download status of a file.
        
        Args:
            file_info: File information dictionary with name and size
            
        Returns:
            Tuple of (status, bytes_downloaded) where status is
            'complete', 'partial', or 'missing'
        """
        filepath = self.cache_dir / file_info['name']
        if not filepath.exists():
            return 'missing', 0
            
        current_size = filepath.stat().st_size
        target_size = file_info['size']
        
        if current_size == target_size:
            return 'complete', current_size
        elif current_size < target_size:
            return 'partial', current_size
        else:
            return 'missing', 0  # Corrupt file
    
    def download_chunks(self, file_info: dict):
        """
        Download a file, yielding progress updates.
        
        Args:
            file_info: File information dictionary with name, size, url
            
        Yields:
            Tuple of (bytes_downloaded, status) where status is 'progress' or 'complete'
            
        Raises:
            requests.exceptions.RequestException: On download failure
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.cache_dir / file_info['name']
        
        status, initial_pos = self.check_file_status(file_info)
        if status == 'complete':
            yield (0, 'complete')
            return
            
        headers = {}   if status != 'partial' else {'Range': f'bytes={initial_pos}-'}
        mode    = 'wb' if status != 'partial' else 'ab'
        
        response = requests.get(file_info['url'], headers=headers, stream=True)
        response.raise_for_status()
        
        with open(filepath, mode) as f:
            for chunk in response.iter_content(chunk_size=self.chunk_size):
                if chunk:
                    f.write(chunk)
                    yield (len(chunk), 'progress')
    
    def fetch_file_listing(self) -> list[dict]:
        """
        Fetch the list of available files from FRDR.
        
        Currently returns a representative list based on the dataset structure.
        The Moisseeva (2020) dataset contains 147 LES wildfire simulations.
        
        Returns:
            List of dictionaries with 'name', 'size', and 'url' for each file
        """
        # TODO: Implement actual web scraping from self.dataset_url when available
        
        file_patterns = [
            ("C1F1R1.nc", 42.3), ("C1F1R2.nc", 41.8), ("C1F2R1.nc", 39.5),
            ("C2F1R1.nc", 44.1), ("C2F1R1_hr1.nc", 12.4), ("C2F1R1_hr2.nc", 12.6),
            ("C3F1R1.nc", 38.7), ("C3F2R1.nc", 40.2),
            ("C4F1R1.nc", 43.9), ("C4F1R2.nc", 42.5),
        ]
        
        base_url = f"{self.dataset_url}/files/" if self.dataset_url else ""
        
        return sorted([
            {'name': name, 'size': int(size_gb * 1e9), 'url': f"{base_url}{name}"}
            for name, size_gb in file_patterns
        ], key=lambda f: f['name'])
    
    @staticmethod
    def format_file_size(size_bytes: float) -> str:
        """
        Format file size in human-readable format.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Formatted string with appropriate unit (GB, MB, KB)
        """
        units = ['TB', 'GB', 'MB', 'KB']
        for i, unit in enumerate(units):
            size = size_bytes / (1024 ** (3 - i))
            if size >= 1:
                return f"{size:.1f} {unit}"
        return f"{size_bytes:.0f} B"
    
    def get_resume_info(self, file_info: dict) -> dict:
        """
        Get resume information for a file download.
        
        Args:
            file_info: File information dictionary
            
        Returns:
            Dictionary with resume details including status and progress
        """
        status, current_size = self.check_file_status(file_info)
        total_size = file_info['size']
        
        return {
            'current_size'     : current_size,
            'progress_percent' : (current_size / total_size * 100) if total_size > 0 else 0,
            'remaining_size'   : total_size - current_size,
            'status'           : status,
            'total_size'       : total_size
        }
    
    def get_undownloaded_files(self, available_files: list[dict]) -> list[dict]:
        """
        Filter available files to only those not yet downloaded.
        
        Args:
            available_files: List of all available files
            
        Returns:
            List of files that need downloading
        """
        existing = self.check_existing_files()
        return [f for f in available_files if f['name'] not in existing]
    
    @staticmethod
    def load_json_manifest(manifest_path: Path) -> dict[str, Any]:
        """
        Load JSON manifest with error handling.
        
        Args:
            manifest_path: Path to the manifest file
            
        Returns:
            Parsed JSON data or empty dict if file doesn't exist
        """
        if not manifest_path.exists():
            return {}
            
        try:
            with open(manifest_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    
    @staticmethod
    def save_json_manifest(data: dict[str, Any], manifest_path: Path) -> bool:
        """
        Save JSON manifest with error handling.
        
        Args:
            data          : Dictionary to save as JSON
            manifest_path : Path to save the manifest
            
        Returns:
            True if successful, False otherwise
        """
        try:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except (IOError, OSError):
            return False
    
    def update_manifest(self, file_info: dict) -> bool:
        """
        Update manifest with newly downloaded file information.
        
        Args:
            file_info: Dictionary with file metadata
            
        Returns:
            True if manifest updated successfully
        """
        manifest = self.load_json_manifest(self.manifest_path)
        manifest.setdefault("files", []).append({
            **file_info,
            "downloaded": datetime.now().isoformat()
        })
        return self.save_json_manifest(manifest, self.manifest_path)