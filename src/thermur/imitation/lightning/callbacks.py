"""
Lightning callbacks for monitoring, evaluation, and visualization.

This module provides PyTorch Lightning callbacks that integrate various systems
into the training loop:
- MonitoringCallback: Integrates metrics collection and event logging
- VisualizationCallback: Provides real-time 3D visualization during training
"""

from __future__        import annotations
from pytorch_lightning import Callback
from typing            import TYPE_CHECKING

if TYPE_CHECKING:
    from pytorch_lightning                 import LightningModule, Trainer
    from pytorch_lightning.utilities.types import STEP_OUTPUT
    from tensordict                        import TensorDictBase
    from thermur.imitation.monitoring      import EventLogger, MetricsCollector
    from thermur.imitation.visualization   import Visualizer


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
            self.collector.update_evaluation_metrics(batch, True)

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
                f"summary/{k}" : v
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


class VisualizationCallback(Callback):
    """
    Real-time 3D visualization callback for training monitoring.

    Provides interactive visualization of the flock simulation during training,
    allowing researchers to observe emergent behaviors, thermal dynamics, and
    control policy evolution in real-time. Automatically logs visualization
    frames to WandB when a logger is present.

    The callback manages the PyVista rendering window lifecycle and ensures
    proper resource cleanup. It only activates in interactive mode to avoid
    blocking headless training runs.
    """

    def __init__(
        self,
        auto_close       : bool              = True,
        fps              : int               = 30,
        start_epoch      : int               = 0,
        update_frequency : int               = 10,
        video_duration   : float             = 30.0,
        visualizer       : Visualizer | None = None,
        watch_run        : bool              = False
    ):
        """
        Configure visualization parameters for training integration.

        Args:
            auto_close       : Automatically close window when training ends
            fps              : Frames per second for video encoding
            start_epoch      : Epoch to start visualization (0 = immediate)
            update_frequency : Update visualization every N batches
            video_duration   : Duration in seconds of each video segment
            visualizer       : Pre-configured Visualizer instance for rendering
            watch_run        : Show live visualization window (from --watch flag)
        """
        super().__init__()
        self.auto_close       = auto_close
        self.fps              = fps
        self.start_epoch      = start_epoch
        self.update_frequency = update_frequency
        self.video_duration   = video_duration
        self.visualizer       = visualizer
        self.watch_run        = watch_run

        # Calculate buffer size from fps and duration
        self.video_buffer_size    = int(fps * video_duration)
        self.batch_counter        = 0
        self.frames_buffer        = []
        self.visualization_active = False

    def _log_video_to_wandb(
        self,
        trainer   : Trainer,
        pl_module : LightningModule
    ):
        """
        Log accumulated frames as video to WandB.

        Encodes buffered frames as MP4 video and uploads to WandB dashboard.
        Clears the frame buffer after successful upload to prevent memory growth.

        Args:
            trainer   : PyTorch Lightning trainer with logger attached
            pl_module : Lightning module for accessing global step
        """
        if not self.frames_buffer or not trainer.logger:
            return
            
        try:
            import wandb
            import numpy as np
            
            video_array = np.array(self.frames_buffer)
            video_data  = {
                "visualization/policy_behavior": wandb.Video(
                    video_array, fps=self.fps, format="mp4"
                ),
                "global_step": trainer.global_step
            }
            
            if experiment := getattr(trainer.logger, 'experiment', None):
                experiment.log(video_data)
            else:
                wandb.log(video_data)
            
            self.frames_buffer.clear()
            
        except Exception:
            pass

    def on_exception(
        self,
        trainer   : Trainer,
        pl_module : LightningModule,
        exception : BaseException
    ):
        """
        Ensure proper cleanup when training fails.

        Closes visualization resources even when training is interrupted
        by an exception, preventing resource leaks and hanging processes.

        Args:
            trainer   : PyTorch Lightning trainer instance
            pl_module : Lightning module being trained
            exception : The exception that interrupted training
        """
        if self.visualizer and self.visualization_active:
            try:
                self.visualizer.close()
            except:
                pass
            finally:
                self.visualization_active = False

    def on_fit_end(
        self,
        trainer   : Trainer,
        pl_module : LightningModule
    ):
        """
        Clean up visualization resources when training completes.

        Logs any remaining frames to WandB and closes any open windows.
        Ensures proper cleanup even if training ends early.

        Args:
            trainer   : PyTorch Lightning trainer that finished
            pl_module : Completed Lightning module
        """
        if not self.visualizer:
            return

        if trainer.logger and self.frames_buffer:
            self._log_video_to_wandb(trainer, pl_module)

        if self.visualization_active and self.auto_close and self.watch_run:
            try:
                self.visualizer.close()
            except Exception:
                pass
            finally:
                self.visualization_active = False

    def on_fit_start(
        self,
        trainer   : Trainer,
        pl_module : LightningModule
    ):
        """
        Initialize visualization when training begins.

        Sets up the rendering context and prepares the visualizer for
        receiving batch data. Automatically prepares for video logging
        when a WandB logger is present.

        Args:
            trainer   : PyTorch Lightning trainer beginning the fit process
            pl_module : Lightning module starting training
        """
        if not self.visualizer:
            return

        if trainer.current_epoch >= self.start_epoch:
            self.visualization_active = True

    def on_train_batch_end(
        self,
        trainer   : Trainer,
        pl_module : LightningModule,
        outputs   : STEP_OUTPUT,
        batch     : TensorDictBase,
        batch_idx : int
    ):
        """
        Update visualization with training batch data.

        Renders the current simulation state if the update frequency
        threshold is met. Captures frames for WandB logging when a logger
        is present and displays live if --watch is enabled.

        Args:
            trainer   : PyTorch Lightning trainer coordinating training
            pl_module : Lightning module processing the batch
            outputs   : Model outputs from the forward pass
            batch     : TensorDict containing current agent states
            batch_idx : Index of the current training batch
        """
        if not self.visualization_active or not self.visualizer:
            return

        self.batch_counter += 1
        if self.batch_counter % self.update_frequency != 0:
            return

        try:
            self.visualizer.update(batch)
            
            if self.watch_run:
                self.visualizer.render()
            
            if trainer.logger and self.visualizer.plotter:
                frame = self.visualizer.plotter.screenshot(return_img=True)
                self.frames_buffer.append(frame)
                
                if len(self.frames_buffer) >= self.video_buffer_size:
                    self._log_video_to_wandb(trainer, pl_module)
                
        except Exception:
            self.visualization_active = False

    def on_train_epoch_end(
        self,
        trainer   : Trainer,
        pl_module : LightningModule
    ):
        """
        Flush visualization buffers at epoch boundaries.

        Ensures any accumulated video frames are logged to WandB before
        moving to the next epoch, preventing data loss.

        Args:
            trainer   : PyTorch Lightning trainer managing epochs
            pl_module : Lightning module completing the epoch
        """
        if trainer.logger and self.frames_buffer:
            self._log_video_to_wandb(trainer, pl_module)

    def on_train_epoch_start(
        self,
        trainer   : Trainer,
        pl_module : LightningModule
    ):
        """
        Activate visualization when the start epoch is reached.

        Allows delayed visualization start to skip early training
        iterations where the policy is still random.

        Args:
            trainer   : PyTorch Lightning trainer tracking epochs
            pl_module : Lightning module beginning new epoch
        """
        if not self.visualizer or self.visualization_active:
            return

        if trainer.current_epoch >= self.start_epoch:
            self.visualization_active = True

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
        Optionally update visualization during validation.

        Shows validation rollouts in the visualization window,
        providing visual confirmation of policy performance on
        held-out data.

        Args:
            trainer        : PyTorch Lightning trainer running validation
            pl_module      : Lightning module being validated
            outputs        : Model outputs from validation forward pass
            batch          : TensorDict containing validation data
            batch_idx      : Index of the current validation batch
            dataloader_idx : Index of the dataloader for multi-dataloader setups
        """
        if not self.visualization_active or not self.visualizer:
            return

        if batch_idx == 0:
            try:
                self.visualizer.update(batch)
                self.visualizer.render()
            except Exception:
                pass