"""
Lightning callbacks for monitoring and evaluation.

This module provides PyTorch Lightning callbacks that integrate various systems
into the training loop:
- MonitoringCallback: Integrates metrics collection and event logging
"""

from __future__        import annotations
from pytorch_lightning import Callback
from typing            import TYPE_CHECKING

if TYPE_CHECKING:
    from pytorch_lightning                 import LightningModule, Trainer
    from pytorch_lightning.utilities.types import STEP_OUTPUT
    from tensordict                        import TensorDictBase
    from thermur.imitation.monitoring      import EventLogger, MetricsCollector


class MonitoringCallback(Callback):
    """
    Unified monitoring callback for metrics and event tracking.

    Consolidates metric collection and event logging into a single callback,
    reducing code duplication and simplifying the training pipeline integration.
    Updates metrics during batch processing and manages state resets at epoch
    boundaries.
    """

    def __init__(
        self,
        collector : MetricsCollector | None = None,
        events    : EventLogger      | None = None
    ):
        """
        Configure monitoring components for training lifecycle integration.

        Args:
            collector : Optional metrics collector for performance tracking
            events    : Optional event logger for tracking critical agent behaviors
        """
        super().__init__()
        self.collector = collector
        self.events    = events

    def on_fit_end(
        self,
        trainer   : Trainer,
        pl_module : LightningModule
    ):
        """
        Clean up resources when training completes.

        Ensures all buffered event data is written to logging backends.

        Args:
            trainer   : PyTorch Lightning trainer coordinating the training process
            pl_module : Lightning module instance
        """
        if self.events:
            self.events.flush_all(pl_module)

    def on_train_batch_end(
        self,
        trainer   : Trainer,
        pl_module : LightningModule,
        outputs   : STEP_OUTPUT,
        batch     : TensorDictBase,
        batch_idx : int
    ):
        """
        Process training batch for metrics and event detection.

        Updates evaluation metrics, tracks CBF activations, and analyzes
        batch data for critical events like thermal violations.

        Args:
            trainer   : PyTorch Lightning trainer managing the training loop
            pl_module : Lightning module containing the policy network
            outputs   : Model outputs from the training step
            batch     : TensorDict containing agent states and environment data
            batch_idx : Index of the current batch within the epoch
        """
        if self.collector:
            # Ensure metrics are on same device as model (minimal fix for device hopping)
            if not hasattr(self, '_metrics_synced'):
                for name in ['train_evaluation', 'val_evaluation', 'train_imitation', 'val_imitation']:
                    metrics = getattr(self.collector, name)
                    setattr(self.collector, name, metrics.to(pl_module.device))
                self._metrics_synced = True
            self.collector.update_evaluation_metrics(batch,  True)

        if self.events:
            self.events.analyze_batch(batch, pl_module)

    def on_train_epoch_end(
        self,
        trainer   : Trainer,
        pl_module : LightningModule
    ):
        """
        Reset per-epoch counters and log summary statistics.

        Clears CBF activation counts and event statistics that are
        tracked on a per-epoch basis for trend analysis. Also logs
        summary metrics for the completed epoch.

        Args:
            trainer   : PyTorch Lightning trainer instance
            pl_module : Lightning module for state management
        """
        if self.events:
            pl_module.log_dict({
                f"events/{k}" : v
                for k, v in self.events.get_event_summary().items()
            })
            self.events.reset_epoch_metrics()

    def on_validation_batch_end(
        self,
        trainer        : Trainer,
        pl_module      : LightningModule,
        outputs        : STEP_OUTPUT,
        batch          : TensorDictBase,
        batch_idx      : int,
        dataloader_idx : int = 0
    ):
        """
        Update metrics and detect events during validation.

        Tracks the same metrics as training but without updating
        model parameters, providing unbiased performance estimates.

        Args:
            trainer        : PyTorch Lightning trainer instance
            pl_module      : Lightning module being validated
            outputs        : Model outputs from the validation step
            batch          : TensorDict containing validation batch data
            batch_idx      : Index of the current validation batch
            dataloader_idx : Index of the dataloader for multi-dataloader setups
        """
        if self.collector:
            self.collector.update_evaluation_metrics(batch, False)

        if self.events:
            self.events.analyze_batch(batch, pl_module)

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