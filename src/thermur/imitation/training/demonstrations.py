"""
Offline demonstration generation and dataset management.

This module provides infrastructure for generating expert demonstrations
offline and serving them efficiently during training via PyG's InMemoryDataset.
"""
from __future__           import annotations
from datetime             import datetime
from hashlib              import sha256
from pathlib              import Path
from torch_geometric.data import InMemoryDataset
from typing               import TYPE_CHECKING

import json
import torch as th

if TYPE_CHECKING:
    from ..controller               import MurmurationController
    from ..simulation               import SimulationEnv
    from config.imitation.training import DemonstrationsModel, ExperienceModel
    from omegaconf                  import DictConfig


class DemonstrationsDataset(InMemoryDataset):
    """
    PyG InMemoryDataset for loading pre-generated demonstrations.
    
    Leverages PyG's built-in caching mechanism to efficiently load
    and serve demonstrations during training. Automatically regenerates
    processed data when configuration changes via hash in filename.
    """
    
    def __init__(self, root: str, config_hash: str):
        """
        Initialize the demonstrations dataset.
        
        Args:
            config_hash : Hash of configuration for cache invalidation
            root        : Root directory containing demonstrations
        """
        self.config_hash = config_hash
        super().__init__(root)
        self.data, self.slices = th.load(self.processed_paths[0])
    
    def process(self):
        """
        Process raw demonstrations into PyG format.
        
        Called automatically by PyG when processed file doesn't exist.
        Loads all demonstration episodes and flattens into single dataset.
        """
        data_list = [
            data
            for episode_file in sorted(Path(self.raw_dir).glob("*.pt"))
            if episode_file.stem != "metadata"
            for data in th.load(episode_file)
        ]
        
        if self.pre_filter:
            data_list = [d for d in data_list if self.pre_filter(d)]
        
        if self.pre_transform:
            data_list = [self.pre_transform(d) for d in data_list]
        
        th.save(self.collate(data_list), self.processed_paths[0])

    @property
    def processed_file_names(self):
        """
        Return dynamic filename based on config hash.
        
        When config changes, filename changes, forcing PyG to reprocess.
        """
        return [f"data_{self.config_hash}.pt"]


class DemonstrationsGenerator:
    """
    Generates and manages offline expert demonstrations.
    
    Converts controller trajectories into demonstrations suitable for
    imitation learning. Handles scenario discovery, incremental generation,
    metadata tracking, and configuration validation.
    
    TODO: Phase 5 - Convert to PyG Data objects:
    - Replace SimulationEnv with minimal physics wrapper
    - State maintained as PyG Data objects throughout
    - Controller.forward(): Data -> action tensor
    - Environment.step(): (Data, action) -> Data
    - Complete removal of TorchRL/TensorDict
    """
    
    def __init__(
        self,
        config     : DemonstrationsModel,
        controller : MurmurationController,
        env        : SimulationEnv,
        experience : ExperienceModel
    ):
        """
        Initialize the demonstrations generator.
        
        Args:
            config     : Demonstrations configuration
            controller : Expert controller for trajectory generation  
            env        : Simulation environment
            experience : Experience model with total_frames target
        """
        super().__init__()
        self.config     = config
        self.controller = controller
        self.env        = env
        self.experience = experience
        self.output_dir = Path("data/demonstrations")
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def _find_wrf_files(self):
        """
        Find WRF data following project's data discovery pattern.
        
        Checks wrf-sfire directory first, then samples, then None for default.
        """
        wrf_sfire = Path("data/wrf-sfire")
        samples   = Path("data/samples")
        
        return (
            sorted(wrf_sfire.glob("*.nc")) if wrf_sfire.exists() else
            sorted(samples.glob("*.nc"))   if samples.exists()   else
            [None]
        )

    def _load_or_create_metadata(self) -> dict:
        """
        Load existing metadata or create new.
        
        Returns metadata dict with scenarios, frame counts, and config hash.
        """
        metadata_file = self.output_dir / "metadata.json"
        
        return (
            json.loads(metadata_file.read_text())
            if metadata_file.exists()
            else {
                "config_hash"  : "",
                "scenarios"    : {},
                "total_frames" : 0,
                "train_split"  : self.config.train_split
            }
        )
    
    def _save_metadata(self, metadata: dict):
        """
        Save metadata to JSON file.
        
        Writes generation statistics and config hash for validation.
        """
        (self.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2)
        )
    
    def generate(self) -> dict:
        """
        Generate demonstrations until experience.total_frames is reached.
        
        Supports incremental generation by loading existing metadata
        and continuing from where previous generation left off.
        
        Returns:
            Metadata dictionary with generation statistics
        """
        metadata  = self._load_or_create_metadata()
        wrf_files = self._find_wrf_files()
        
        while metadata["total_frames"] < self.experience.total_frames:
            for wrf_file in wrf_files:
                if metadata["total_frames"] >= self.experience.total_frames:
                    break
                
                scenario_name = wrf_file.stem if wrf_file else "default"
                episode_num   = metadata["scenarios"].setdefault(
                    scenario_name, {"episodes": 0, "frames": 0}
                )["episodes"]
                
                # TODO: Properly set WRF file when implementing CLI command (Phase 4)
                # Need to either:
                # 1. Recreate environment with new WRF file
                # 2. Add method to environment to switch data sources
                # 3. Pass WRF file to generate_trajectories directly
                # if wrf_file:
                #     self.env.loader.data_path = str(wrf_file)
                
                th.save(
                    self.controller.generate_trajectories(
                        env           = self.env,
                        num_timesteps = self.config.frames_per_episode
                    ),
                    self.output_dir / f"{scenario_name}_ep{episode_num:04d}.pt"
                )
                
                metadata["scenarios"][scenario_name]["episodes"] += 1
                metadata["scenarios"][scenario_name]["frames"]   += self.config.frames_per_episode
                metadata["total_frames"]                         += self.config.frames_per_episode
        
        metadata["generated"] = datetime.now().isoformat()
        self._save_metadata(metadata)
        
        return metadata
    
    def get_config_hash(self, cfg: DictConfig) -> str:
        """
        Generate hash of configuration parameters that affect demonstrations.
        
        Includes data source in hash so sample vs full data have different caches.
        
        Args:
            cfg : Full configuration object
        
        Returns:
            16-character hash string
        """
        wrf_files = self._find_wrf_files()
        data_source = "sample" if any("sample" in str(f) for f in wrf_files if f) else "full"
        
        return sha256(
            json.dumps({
                "configs"     : {k: dict(v) for k, v in cfg.items() if k != "hardware"},
                "data_source" : data_source
            }, sort_keys=True).encode()
        ).hexdigest()[:16]
    
    def validate_config(self, cfg: DictConfig) -> bool:
        """
        Check if existing demonstrations match current configuration.
        
        Args:
            cfg : Current configuration
        
        Returns:
            True if demonstrations are valid for current config
        """
        metadata_file = self.output_dir / "metadata.json"
        
        return (
            metadata_file.exists() and
            json.loads(metadata_file.read_text()).get("config_hash") == 
            self.get_config_hash(cfg)
        )