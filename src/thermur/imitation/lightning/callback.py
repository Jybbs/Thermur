"""
Lightning callbacks for monitoring and evaluation.

This module provides a unified PyTorch Lightning callback that integrates the
monitoring system (MetricsCollector and EventLogger) into the training loop,
handling metric updates and event logging at appropriate lifecycle hooks.
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
        Flush accumulated events and log final summary statistics.

        Ensures all buffered event data is written to logging backends
        and provides aggregate statistics for the entire training run.
        """
        if not self.events:
            return

        self.events.flush_all(pl_module)
        pl_module.log_dict({
            f"summary/{k}": v
            for k, v in self.events.get_event_summary().items()
        })

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
        """
        if self.collector:
            self.collector.update_evaluation_metrics(batch, True)

        if self.events:
            self.events.analyze_batch(batch, pl_module)

    def on_train_epoch_end(
        self,
        trainer   : Trainer,
        pl_module : LightningModule
    ):
        """
        Reset per-epoch counters to ensure accurate rate calculations.

        Clears CBF activation counts and event statistics that are
        tracked on a per-epoch basis for trend analysis.
        """
        if self.events:
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
        """
        if self.collector:
            self.collector.log_all_metrics(
                is_training = False,
                module      = pl_module,
                step_data   = None
            )
