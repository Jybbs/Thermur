"""
Offline expert dataset with automatic caching.

Provides a PyG InMemoryDataset that generates and caches expert trajectories
for behavioral cloning. Automatically regenerates when configuration changes
via hash-based filenames. Uses stratified sampling to ensure equal
representation across all WRF snapshots.
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


class ExpertDataset(InMemoryDataset):
    """
    PyG InMemoryDataset for expert trajectories.

    Generates trajectories on first access and caches them. Automatically
    regenerates when configuration changes via hash-based filenames. Uses
    stratified sampling across WRF snapshots for balanced dataset diversity.

    Registers PyG Data class as safe for PyTorch 2.6+ compatibility.
    """

    def __init__(
        self,
        controller                : DictConfig,
        environment               : DictConfig,
        generator                 : TrajectoryGenerator,
        murmuration               : MurmurationController,
        sample_url                : str,
        trajectories_per_snapshot : int,
        trajectory_duration       : float,
        ui                        : ThermurUI
    ):
        """
        Initialize the expert dataset.

        Args:
            controller                : Controller configuration
            environment               : Environment configuration
            generator                 : Trajectory generator for physics simulation
            murmuration               : Expert controller for demonstrations
            sample_url                : URL for downloading sample WRF dataset
            trajectories_per_snapshot : Number of trajectories per 15s WRF snapshot
            trajectory_duration       : Duration of each trajectory in seconds
            ui                        : CLI UI instance for progress display
        """
        self.controller                = controller
        self.environment               = environment
        self.generator                 = generator
        self.hash                      = None
        self.murmuration               = murmuration
        self.sample_url                = sample_url
        self.trajectories_per_snapshot = trajectories_per_snapshot
        self.trajectory_duration       = trajectory_duration
        self.ui                        = ui
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
        Factory method creating a PyTorch Lightning DataModule with automatic
        train/val splitting and configuration-based cache invalidation.

        The resulting LightningDataset handles batching, shuffling, and
        multi-GPU distribution automatically.

        Args:
            batch_size  : Number of graph states per training batch
            controller  : Controller configuration
            environment : Environment configuration
            generator   : Trajectory generator for physics simulation
            hardware    : Hardware configuration for dataloader settings
            murmuration : Expert controller for demonstrations
            train_split : Fraction of data reserved for training
            ui          : CLI UI instance for progress display

        Returns:
            LightningDataset configured with train/val splits
        """
        dataset = cls(
            controller                = controller,
            environment               = environment,
            generator                 = generator,
            murmuration               = murmuration,
            sample_url                = environment.dataset.sample_url,
            trajectories_per_snapshot = environment.dataset.trajectories_per_snapshot,
            trajectory_duration       = environment.dataset.trajectory_duration,
            ui                        = ui
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
        Generate demonstrations using stratified sampling across snapshots.

        Called automatically by PyG when processed file does not exist.
        Generates expert trajectories with consistent environmental conditions
        by sampling each WRF snapshot equally, ensuring dataset diversity.
        """
        self.generator.wrf.load_datasets(self.raw_paths)

        trajectory_frames  = int(
            self.trajectory_duration / self.generator.physics.timeframe
        )
        n_snapshots        = self.generator.wrf.n_snapshots
        total_trajectories = n_snapshots * self.trajectories_per_snapshot
        total_frames       = total_trajectories * trajectory_frames

        with self.ui.create_thermal_progress() as progress:
            task = progress.add_task(
                description = "Generating expert trajectories",
                total       = total_frames
            )

            data_list = []
            for snapshot_idx in range(n_snapshots):
                for _ in range(self.trajectories_per_snapshot):
                    data_list.extend(
                        self.murmuration.generate_trajectory(
                            generator    = self.generator,
                            num_frames   = trajectory_frames,
                            snapshot_idx = snapshot_idx
                        )
                    )
                    progress.update(task, advance=trajectory_frames)

        self.ui.print_message("Saving expert trajectories to cache...", "info")
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
