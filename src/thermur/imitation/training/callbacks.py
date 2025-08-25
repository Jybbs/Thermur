"""
Lightning callbacks for monitoring and evaluation.

This module provides PyTorch Lightning callbacks that integrate various systems
into the training loop:
- MonitoringCallback: Integrates metrics collection
"""

from __future__        import annotations
from pytorch_lightning import Callback
from typing            import TYPE_CHECKING

if TYPE_CHECKING:
    from .metrics                          import MetricsCollector
    from pytorch_lightning                 import LightningModule, Trainer
    from pytorch_lightning.utilities.types import STEP_OUTPUT
    from torch_geometric.data              import Batch


class MonitoringCallback(Callback):
    """
    Unified monitoring callback for metrics tracking.

    Consolidates metric collection into a single callback, reducing code
    duplication and simplifying the training pipeline integration. Updates
    metrics during batch processing and manages state resets at epoch boundaries.
    """

    def __init__(
        self,
        collector : MetricsCollector | None = None
    ):
        """
        Configure monitoring components for training lifecycle integration.

        Args:
            collector : Optional metrics collector for performance tracking
        """
        super().__init__()
        self.collector = collector

    def on_fit_end(
        self,
        trainer   : Trainer,
        pl_module : LightningModule
    ):
        """
        Clean up resources when training completes.

        Args:
            trainer   : PyTorch Lightning trainer coordinating the training process
            pl_module : Lightning module instance
        """
        pass

    def on_train_batch_end(
        self,
        trainer   : Trainer,
        pl_module : LightningModule,
        outputs   : STEP_OUTPUT,
        batch     : Batch,
        batch_idx : int
    ):
        """
        Process training batch for metrics.

        Updates evaluation metrics from the training batch.

        Args:
            trainer   : PyTorch Lightning trainer managing the training loop
            pl_module : Lightning module containing the policy network
            outputs   : Model outputs from the training step
            batch     : PyG Batch containing agent states and environment data
            batch_idx : Index of the current batch within the epoch
        """
        if self.collector:

            if not hasattr(self, '_metrics_synced'):
                for name in ['train_evaluation', 'val_evaluation', 'train_imitation', 'val_imitation']:
                    metrics = getattr(self.collector, name)
                    setattr(self.collector, name, metrics.to(pl_module.device))
                self._metrics_synced = True
            self.collector.update_evaluation_metrics(batch,  True)

    def on_train_epoch_end(
        self,
        trainer   : Trainer,
        pl_module : LightningModule
    ):
        """
        Reset per-epoch counters and log summary statistics.

        Logs summary metrics for the completed epoch.

        Args:
            trainer   : PyTorch Lightning trainer instance
            pl_module : Lightning module for state management
        """
        pass

    def on_validation_batch_end(
        self,
        trainer        : Trainer,
        pl_module      : LightningModule,
        outputs        : STEP_OUTPUT,
        batch          : Batch,
        batch_idx      : int,
        dataloader_idx : int = 0
    ):
        """
        Update metrics during validation.

        Tracks the same metrics as training but without updating
        model parameters, providing unbiased performance estimates.

        Args:
            trainer        : PyTorch Lightning trainer instance
            pl_module      : Lightning module being validated
            outputs        : Model outputs from the validation step
            batch          : PyG Batch containing validation batch data
            batch_idx      : Index of the current validation batch
            dataloader_idx : Index of the dataloader for multi-dataloader setups
        """
        if self.collector:
            self.collector.update_evaluation_metrics(batch, False)

    def on_validation_epoch_end(
        self,
        trainer   : Trainer,
        pl_module : LightningModule
    ):
        """
        Aggregate and log validation metrics for epoch-level tracking.

        Computes final metric values across all validation batches
        for monitoring training progress and early stopping decisions.

        Args:
            trainer   : PyTorch Lightning trainer managing validation
            pl_module : Lightning module containing metrics to log
        """
        if self.collector:
            self.collector.log_all_metrics(
                is_training = False,
                module      = pl_module,
                step_data   = None
            )