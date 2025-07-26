"""
Lightning callbacks for monitoring and evaluation.

This module provides a unified PyTorch Lightning callback that integrates the
monitoring system (MetricsCollector and EventLogger) into the training loop,
handling metric updates and event logging at appropriate lifecycle hooks.
"""
from pytorch_lightning            import Callback, LightningModule, Trainer
from tensordict                   import TensorDict
from thermur.imitation.monitoring import EventLogger, MetricsCollector
from typing                       import Any, Optional


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
        events  : Optional[EventLogger]      = None,
        metrics : Optional[MetricsCollector] = None
    ):
        """
        Configure monitoring components for training lifecycle integration.
        
        Args:
            events  : Optional event logger for tracking critical agent behaviors
            metrics : Optional metrics collector for performance tracking
        """
        super().__init__()
        self.events  = events
        self.metrics = metrics
    
    def on_fit_end(
        self, 
        _trainer  : Trainer,
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
        batch      : TensorDict,
        pl_module  : LightningModule,
        _batch_idx : int,
        _outputs   : Any,
        _trainer   : Trainer
    ):
        """
        Process training batch for metrics and event detection.
        
        Updates evaluation metrics, tracks CBF activations, and analyzes
        batch data for critical events like thermal violations.
        """
        if self.metrics:
            self.metrics.update_evaluation_metrics(batch, "train")
            self.metrics.log_cbf_activation(batch)
            
        if self.events:
            self.events.analyze_batch(batch, pl_module)
    
    def on_train_epoch_end(
        self, 
        _pl_module : LightningModule,
        _trainer   : Trainer 
    ):
        """
        Reset per-epoch counters to ensure accurate rate calculations.
        
        Clears CBF activation counts and event statistics that are
        tracked on a per-epoch basis for trend analysis.
        """
        if self.metrics:
            self.metrics.reset_runtime_metrics()
            
        if self.events:
            self.events.reset_epoch_metrics()
    
    def on_validation_batch_end(
        self,
        batch      : TensorDict,
        pl_module  : LightningModule,
        _batch_idx : int,
        _outputs   : Any,
        _trainer   : Trainer
    ):
        """
        Update metrics and detect events during validation.
        
        Tracks the same metrics as training but without updating
        model parameters, providing unbiased performance estimates.
        """
        if self.metrics:
            self.metrics.update_evaluation_metrics(batch, "val")
            
        if self.events:
            self.events.analyze_batch(batch, pl_module)
    
    def on_validation_epoch_end(
        self, 
        pl_module : LightningModule,
        _trainer  : Trainer 
    ):
        """
        Aggregate and log validation metrics for epoch-level tracking.
        
        Computes final metric values across all validation batches
        for monitoring training progress and early stopping decisions.
        """
        if self.metrics:
            self.metrics.log_all_metrics(
                loss   = None,
                module = pl_module,
                phase  = "val"
            )