"""
Offline demonstration dataset with automatic caching.

Provides a PyG InMemoryDataset that generates and caches expert demonstrations.
Automatically regenerates when configuration changes via hash-based filenames.
"""
from __future__                     import annotations
from hashlib                        import sha256
from pathlib                        import Path
from tarfile                        import open as tf_open
from torch_geometric.data           import InMemoryDataset
from torch_geometric.data.lightning import LightningDataset
from typing                         import TYPE_CHECKING
from urllib.request                 import urlretrieve

import pickle as pk
import torch  as th

if TYPE_CHECKING:
    from ..controller              import MurmurationController
    from ..simulation              import SimulationEnv
    from config.imitation.training import DemonstrationsModel, HardwareModel
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
        
        super().__init__("data")
        self.data, self.slices = th.load(self.processed_paths[0])
    
    def _compute_hash(self) -> str:
        """
        Generate hash of configuration parameters affecting demonstrations.
        
        Uses hashable dict plus count of loaded WRF datasets for cache invalidation.
        """
        container = dict(self.hashable)
        container["num_datasets"] = len(self.env.wrf.datasets)
        
        return sha256(pk.dumps(container)).hexdigest()[:16]
    
    @staticmethod
    def _find_netcdf_files() -> list[str]:
        """
        Find NetCDF files by checking magic numbers.
        
        NetCDF files start with 'CDF' or '\x89HDF' magic bytes.
        """
        raw_dir = Path("data/raw")
        if not raw_dir.exists():
            return []
        
        def is_netcdf(path):
            """
            Check if file has NetCDF magic bytes.
            """
            try:
                with open(path, 'rb') as f:
                    magic = f.read(4)
                    return magic[:3] == b'CDF' or magic == b'\x89HDF'
            except Exception:
                return False
        
        return [
            file.relative_to(raw_dir).as_posix()
            for file in raw_dir.rglob("*")
            if file.is_file() and is_netcdf(file)
        ]
    
    @classmethod
    def as_lightning_datamodule(
        cls,
        controller     : MurmurationController,
        demonstrations : DemonstrationsModel,
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
    
    def download(self):
        """
        Auto-download sample dataset if no NetCDF files found.
        
        Called by PyG when files in raw_file_names don't exist.
        """
        self.ui.print_message(
            "No WRF data found. Downloading sample dataset...", "info"
        )
        
        (sample_tar := Path(self.raw_dir) / "samples.tar.gz").parent.mkdir(
            exist_ok = True, 
            parents  = True
        )
        
        with self.ui.create_thermal_progress() as progress:
            task = progress.add_task(
                description = "Downloading sample WRF data",
                total       = 100
            )
            
            reporthook = lambda block_num, block_size, total_size: progress.update(
                completed = min(100 * block_num * block_size / total_size, 100),
                task_id   = task 
            )
            
            try:
                urlretrieve(
                    filename   = sample_tar, 
                    reporthook = reporthook,
                    url        = self.demonstrations.sample_url
                )
                
                progress.update(task, description="Extracting sample data...")
                with tf_open(sample_tar, 'r:gz') as tar:
                    tar.extractall(self.raw_dir)
                
                sample_tar.unlink()
                progress.update(
                    completed   = 100, 
                    description = "Sample dataset ready!",
                    task_id     = task)
                
            except Exception as e:
                raise FileNotFoundError(
                    f"Failed to download sample dataset: {e}\n"
                    f"Please manually download WRF data to data/raw/"
                ) from e

    
    def process(self):
        """
        Generate demonstrations and save to cache.
        
        Called automatically by PyG when processed file doesn't exist.
        Generates expert trajectories across WRF scenarios until
        total_frames reached.
        """
        frames_per_ep  = self.demonstrations.frames_per_episode
        total_episodes = self.demonstrations.total_frames // frames_per_ep
        
        with self.ui.create_thermal_progress() as progress:
            task = progress.add_task(
                description = "Generating expert demonstrations", 
                total       = self.demonstrations.total_frames
            )
            
            data_list = []
            for _ in range(total_episodes):
                data_list.extend(
                    self.controller.generate_trajectories(self.env, frames_per_ep)
                )
                progress.update(task, advance=frames_per_ep)
        
        self.ui.print_message("Saving demonstrations to cache...", "info")
        th.save(self.collate(data_list), self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        """
        Dynamic filename based on config hash for automatic cache invalidation.
        """
        return [f"data_{self.config_hash}.pt"]
    
    @property
    def raw_file_names(self):
        """
        Return list of NetCDF files found in raw directory.
        
        PyG checks if these exist before calling download().
        """
        return self._find_netcdf_files() or ["samples/wrf_sample.nc"]
