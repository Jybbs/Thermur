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
    from thermur.cli.helpers       import ThermurUI


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
        ui             : ThermurUI,
        root           : str | None = None
    ):
        """
        Initialize the demonstrations dataset.
        
        Args:
            cfg            : Full configuration for hash computation
            controller     : Expert controller for trajectory generation
            demonstrations : Demonstrations configuration
            env            : Simulation environment
            ui             : CLI UI instance for progress display
            root           : Cache directory (auto-determined if None)
        """
        self.cfg            = cfg
        self.config_hash    = self._compute_hash(cfg)
        self.controller     = controller
        self.demonstrations = demonstrations
        self.env            = env
        self.ui             = ui
        
        if root is None:
            wrf_files = self._find_wrf_files()
            if not wrf_files:
                raise FileNotFoundError(
                    "No WRF data files found. Run 'thermur download -s' to get "
                    "started with sample data."
                )
            source_dir = wrf_files[0].parent
        
        super().__init__(str(source_dir).replace("/raw/", "/processed/"))
        self.data, self.slices = th.load(self.processed_paths[0])
    
    def process(self):
        """
        Generate demonstrations and save to cache.
        
        Called automatically by PyG when processed file doesn't exist.
        Generates expert trajectories across WRF scenarios until total_frames reached.
        """
        # TODO: Phase 4 - Use wrf_files to vary environments
        # wrf_files = self._find_wrf_files()
        # Could use itertools slice/cycle
        
        frames_per_ep  = self.demonstrations.frames_per_episode
        total_episodes = self.demonstrations.total_frames // frames_per_ep
        data_list      = []
        
        with self.ui.create_thermal_progress() as progress:
            task = progress.add_task(
                description = "Generating expert demonstrations", 
                total       = self.demonstrations.total_frames
            )
            
            for _ in range(total_episodes):
                trajectory = self.controller.generate_trajectories(
                    env           = self.env,
                    num_timesteps = frames_per_ep
                )
                
                data_list.extend(trajectory)
                progress.update(task, advance=frames_per_ep)
        
        self.ui.print_message("Saving demonstrations to cache...", "info")
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
        wrf_sfire = Path(self.cfg.cli.download.wrf_sfire_dir)
        sample    = Path(self.cfg.cli.download.sample_data_path)
        
        if wrf_sfire.exists():
            files = sorted(wrf_sfire.glob("*.nc"))
            if files:
                return files
        
        return [sample] if sample.exists() else []
