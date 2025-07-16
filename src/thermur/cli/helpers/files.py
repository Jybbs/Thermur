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


class FileIO:
    """
    Manages file I/O operations for the CLI.
    
    Provides utilities for JSON manifest operations, file downloads,
    size formatting, and other common file-related tasks with proper
    error handling.
    """
    
    def __init__(self, cache_dir: Path):
        """
        Initialize the file I/O helper for manifest and cache management.
        
        Args:
            cache_dir : Directory for caching downloaded files and metadata
        """
        self.cache_dir     = cache_dir
        self.manifest_path = cache_dir / "manifest.json"
    
    def check_existing_files(self) -> set[str]:
        """
        Check manifest and cache directory for already downloaded files.
        
        Returns:
            Set of filenames that have already been downloaded
        """
        manifest_files = {f["name"] for f in self.load_json_manifest(self.manifest_path).get("files", [])}
        disk_files     = {f.name for f in self.cache_dir.glob("*.nc")} if self.cache_dir.exists() else set()
        return manifest_files | disk_files
    
    
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