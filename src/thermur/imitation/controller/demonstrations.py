"""
Offline demonstration dataset with automatic caching.

Provides a PyG InMemoryDataset that generates and caches expert demonstrations.
Automatically regenerates when configuration changes via hash-based filenames.
"""
from __future__                     import annotations
from hashlib                        import file_digest, sha256
from pathlib                        import Path
from json                           import dumps
from omegaconf                      import OmegaConf
from torch_geometric.data           import Data, InMemoryDataset, extract_tar
from torch_geometric.data.lightning import LightningDataset
from typing                         import TYPE_CHECKING
from urllib.request                 import urlretrieve

import torch as th

if TYPE_CHECKING:
    from ..environment             import TrajectoryGenerator
    from .murmuration              import MurmurationController
    from config.imitation.training import HardwareModel
    from omegaconf                 import DictConfig
    from thermur.cli.helpers       import ThermurUI


class DemonstrationsDataset(InMemoryDataset):
    """
    PyG InMemoryDataset for expert demonstrations.

    Generates demonstrations on first access and caches them. Automatically
    regenerates when configuration changes via hash-based filename.

    Registers PyG Data class as safe for PyTorch 2.6+ compatibility.
    """

    def __init__(
        self,
        controller         : DictConfig,
        environment        : DictConfig,
        frames_per_episode : int,
        generator          : TrajectoryGenerator,
        murmuration        : MurmurationController,
        sample_url         : str,
        total_frames       : int,
        ui                 : ThermurUI
    ):
        """
        Initialize the demonstrations dataset.

        Args:
            controller         : Controller configuration for cache invalidation
            environment        : Environment configuration for cache invalidation
            frames_per_episode : Number of timesteps per demonstration episode
            generator          : Trajectory generator for physics simulation
            murmuration        : Expert controller for trajectory generation
            sample_url         : URL for downloading sample WRF dataset
            total_frames       : Total demonstration frames to generate
            ui                 : CLI UI instance for progress display
        """
        self.controller         = controller
        self.environment        = environment
        self.frames_per_episode = frames_per_episode
        self.generator          = generator
        self.hash               = None
        self.murmuration        = murmuration
        self.sample_url         = sample_url
        self.total_frames       = total_frames
        self.ui                 = ui
        th.serialization.add_safe_globals([Data])

        super().__init__("data")
        self.load(self.processed_paths[0])
        self._make_picklable()

    def _compute_hash(self):
        """
        Generate deterministic hash of configuration parameters.

        Converts DictConfigs to primitive containers with resolved values,
        adds WRF file checksums, then uses JSON serialization with sorted
        keys to ensure deterministic output before hashing.
        """
        to_container = lambda c: OmegaConf.to_container(c, resolve=True)
        self.hash    = self.hash or sha256(dumps(
            {
                "controller"  : to_container(self.controller),
                "environment" : to_container(self.environment),
                "checksums"   : {
                    f"wrf_{i}":
                    file_digest(open(path, 'rb'), 'sha256').hexdigest()[:8]
                    for i, path in enumerate(self.raw_paths)
                }
            },
            default   = lambda o: o.model_dump(),
            sort_keys = True
        ).encode()).hexdigest()[:16]

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

    def _make_picklable(self):
        """
        Remove unpicklable attributes after successful data load.

        Removes known unpicklable types by checking class names.
        This ensures the dataset can be used with multiprocessing after
        cache loading, while preserving all picklable configuration.
        """
        [
            delattr(self, attr) for attr in list(self.__dict__)
            if type(getattr(self, attr)).__name__ in
            {
                'DictConfig',
                'MurmurationController', 'ThermurUI', 'TrajectoryGenerator'
            }
        ]

    @classmethod
    def as_lightning_datamodule(
        cls,
        batch_size  : int,
        controller  : DictConfig,
        environment : DictConfig,
        generator   : TrajectoryGenerator,
        hardware    : HardwareModel,
        murmuration : MurmurationController,
        train_split : float,
        ui          : ThermurUI
    ) -> LightningDataset:
        """
        Factory method that creates a PyTorch Lightning DataModule with automatic
        train/val splitting, first-time generation detection, and configuration-based
        cache invalidation. The resulting LightningDataset handles all batching,
        shuffling, and multi-GPU distribution automatically.

        Args:
            batch_size  : Number of graph snapshots per training batch
            controller  : Controller configuration
            environment : Environment configuration
            generator   : Trajectory generator for physics simulation
            hardware    : Hardware configuration for dataloader settings
            murmuration : Expert controller for trajectory generation
            train_split : Fraction of data reserved for training
            ui          : CLI UI instance for progress display

        Returns:
            LightningDataset configured with train/val splits
        """
        dataset = cls(
            controller         = controller,
            environment        = environment,
            frames_per_episode = controller.mmm.frames_per_episode,
            generator          = generator,
            murmuration        = murmuration,
            sample_url         = environment.loader.sample_url,
            total_frames       = controller.mmm.total_frames,
            ui                 = ui
        )

        train_size = int(len(dataset) * train_split)
        indices    = th.randperm(len(dataset))

        return LightningDataset(
            batch_size    = batch_size,
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
                    url        = self.sample_url
                )

                progress.update(task, description="Extracting sample data...")
                extract_tar(str(sample_tar), self.raw_dir)

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
        self.generator.wrf.load_datasets(self.raw_paths)

        with self.ui.create_thermal_progress() as progress:
            task = progress.add_task(
                description = "Generating expert demonstrations",
                total       = self.total_frames
            )

            data_list = []
            for _ in range(self.total_frames // self.frames_per_episode):
                data_list.extend(
                    self.murmuration.generate_trajectories(
                        generator     = self.generator,
                        num_timesteps = self.frames_per_episode
                    )
                )
                progress.update(task, advance=self.frames_per_episode)

        self.ui.print_message("Saving demonstrations to cache...", "info")
        self.save(data_list, self.processed_paths[0])

    @property
    def processed_file_names(self):
        """
        Dynamic filename based on config hash for automatic cache invalidation.
        """
        self._compute_hash()
        return [f"{self.hash}/data.pt"]

    @property
    def raw_file_names(self):
        """
        Return list of NetCDF files found in raw directory.

        PyG checks if these exist before calling download().
        """
        return self._find_netcdf_files() or ["samples/wrf_sample.nc"]
