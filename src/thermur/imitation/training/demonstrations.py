"""
Offline demonstration dataset with automatic caching.

Provides a PyG InMemoryDataset that generates and caches expert demonstrations.
Automatically regenerates when configuration changes via hash-based filenames.
"""
from __future__                     import annotations
from hashlib                        import sha256
from omegaconf                      import OmegaConf
from pathlib                        import Path
from torch_geometric.data           import InMemoryDataset
from torch_geometric.data.lightning import LightningDataset
from typing                         import TYPE_CHECKING

import pickle as pk
import torch  as th

if TYPE_CHECKING:
    from ..controller              import MurmurationController
    from ..simulation              import SimulationEnv
    from config.imitation.training import DemonstrationsModel, HardwareModel
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
        controller     : MurmurationController,
        demonstrations : DemonstrationsModel,
        env            : SimulationEnv,
        hashable       : dict,
        ui             : ThermurUI
    ):
        """
        Initialize the demonstrations dataset.
        
        Args:
            controller     : Expert controller for trajectory generation
            demonstrations : Demonstrations configuration
            env            : Simulation environment
            hashable       : Configuration dict for cache invalidation
            ui             : CLI UI instance for progress display
        """
        self.controller     = controller
        self.demonstrations = demonstrations
        self.env            = env
        self.hashable       = hashable
        self.ui             = ui
        self.config_hash    = self._compute_hash()
        
        super().__init__(self._get_processed_root())
        self.data, self.slices = th.load(self.processed_paths[0])
    
    def _compute_hash(self) -> str:
        """
        Generate hash of configuration parameters affecting demonstrations.
        
        Uses hashable dict plus WRF data sources for cache invalidation.
        """
        container = dict(self.hashable)
        container["data_source"] = [f.name for f in self._find_wrf_files()]
        
        return sha256(pk.dumps(container)).hexdigest()[:16]
    
    def _find_wrf_files(self) -> list[Path]:
        """
        Find WRF data following project's discovery pattern.
        
        Checks wrf-sfire first for full dataset, then falls back to sample.
        """
        wrf_sfire = Path(self.download_paths.wrf_sfire_dir)
        sample    = Path(self.download_paths.sample_data_path)
        
        if wrf_sfire.exists():
            files = sorted(wrf_sfire.glob("*.nc"))
            if files:
                return files
        
        return [sample] if sample.exists() else []
    
    def _get_processed_root(self) -> str:
        """
        Determine the processed data root directory based on available WRF files.
        
        Returns:
            Path to processed data directory
            
        Raises:
            FileNotFoundError: If no WRF data files are found
        """
        if not (wrf_files := self._find_wrf_files()):
            raise FileNotFoundError(
                "No WRF data files found. Run 'thermur download -s' "
                "to get started with sample data."
            )
        return str(wrf_files[0].parent).replace("/raw/", "/processed/")
    
    @classmethod
    def as_lightning_datamodule(
        cls,
        controller     : MurmurationController,
        demonstrations : DemonstrationsModel,
        download_paths : DownloadModel,
        env            : SimulationEnv,
        hardware       : HardwareModel,
        hashable       : dict,
        ui             : ThermurUI
    ) -> LightningDataset:
        """
        Factory method that creates a PyTorch Lightning DataModule with automatic
        train/val splitting, first-time generation detection, and configuration-based
        cache invalidation. The resulting LightningDataset handles all batching,
        shuffling, and multi-GPU distribution automatically.
        
        Args:
            controller     : Expert controller for trajectory generation
            demonstrations : Demonstrations configuration with train_split
            download_paths : Download paths for WRF data discovery
            env            : Simulation environment
            hardware       : Hardware configuration for dataloader settings
            hashable       : Configuration dict for cache invalidation
            ui             : CLI UI instance for progress display
        
        Returns:
            LightningDataset configured with train/val splits
        """
        dataset = cls(
            controller     = controller,
            demonstrations = demonstrations,
            download_paths = download_paths,
            env            = env,
            hashable       = hashable,
            ui             = ui
        )
        
        if not Path(dataset.processed_paths[0]).exists():
            ui.print_message(
                "Generating expert demonstrations for the first time. "
                "This will be cached for future runs.",
                "info"
            )
        
        train_size = int(len(dataset) * demonstrations.train_split)
        indices    = th.randperm(len(dataset))
        
        return LightningDataset(
            batch_size    = demonstrations.batch_size,
            num_workers   = hardware.num_workers,
            pin_memory    = hardware.pin_memory,
            train_dataset = dataset.index_select(indices[:train_size]),
            val_dataset   = dataset.index_select(indices[train_size:]),
        )
    
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
