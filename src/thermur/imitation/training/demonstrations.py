"""
Offline demonstration dataset with automatic caching.

Provides a PyG InMemoryDataset that generates and caches expert demonstrations.
Automatically regenerates when configuration changes via hash-based filenames.
"""
from __future__           import annotations
from hashlib              import sha256
from omegaconf            import OmegaConf
from pathlib              import Path
from torch_geometric.data import InMemoryDataset
from typing               import TYPE_CHECKING

import pickle as pk
import torch  as th

if TYPE_CHECKING:
    from ..controller              import MurmurationController
    from ..simulation              import SimulationEnv
    from config.imitation.training import DemonstrationsModel
    from omegaconf                 import DictConfig


class DemonstrationsDataset(InMemoryDataset):
    """
    PyG InMemoryDataset for expert demonstrations.
    
    Generates demonstrations on first access and caches them. Automatically
    regenerates when configuration changes via hash-based filename.
    """
    
    def __init__(
        self,
        cfg            : DictConfig,
        controller     : MurmurationController,
        demonstrations : DemonstrationsModel,
        env            : SimulationEnv,
        root           : str = "data/demonstrations"
    ):
        """
        Initialize the demonstrations dataset.
        
        Args:
            cfg            : Full configuration for hash computation
            controller     : Expert controller for trajectory generation
            demonstrations : Demonstrations configuration
            env            : Simulation environment
            root           : Cache directory (default: data/demonstrations)
        """
        self.cfg            = cfg
        self.config_hash    = self._compute_hash(cfg)
        self.controller     = controller
        self.demonstrations = demonstrations
        self.env            = env
        super().__init__(root)
        self.data, self.slices = th.load(self.processed_paths[0])
    
    def process(self):
        """
        Generate demonstrations and save to cache.
        
        Called automatically by PyG when processed file doesn't exist.
        Generates expert trajectories across WRF scenarios until total_frames reached.
        """
        data_list = []
        total     = 0
        wrf_files = self._find_wrf_files()
        
        while total < self.demonstrations.total_frames:
            for wrf in wrf_files:
                if total >= self.demonstrations.total_frames:
                    break
                    
                # TODO: Phase 4 - Switch environment to use wrf file
                # For now, generate with current environment settings
                trajectory = self.controller.generate_trajectories(
                    env           = self.env,
                    num_timesteps = self.demonstrations.frames_per_episode
                )
                
                data_list.extend(trajectory)
                total += self.demonstrations.frames_per_episode
        
        th.save(self.collate(data_list), self.processed_paths[0])

    @property
    def processed_file_names(self):
        """
        Dynamic filename based on config hash for automatic cache invalidation.
        """
        return [f"data_{self.config_hash}.pt"]
    
    def _compute_hash(self, cfg: DictConfig) -> str:
        """
        Generate hash of configuration parameters affecting demonstrations.
        
        Excludes hardware config, includes WRF data sources.
        """
        container = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(container, dict)
        container.pop("hardware", None)
        
        container["data_source"] = [f.name for f in self._find_wrf_files()]
        
        return sha256(pk.dumps(container)).hexdigest()[:16]
    
    def _find_wrf_files(self) -> list[Path]:
        """
        Find WRF data following project's discovery pattern.
        
        Checks wrf-sfire first for full dataset, then falls back to sample.
        """
        wrf_sfire = Path("data/wrf-sfire")
        sample    = Path("data/samples/wrf_sample.nc")
        
        if wrf_sfire.exists():
            files = sorted(wrf_sfire.glob("*.nc"))
            if files:
                return files
        
        return [sample] if sample.exists() else []
