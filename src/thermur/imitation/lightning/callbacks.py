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


class VisualizationCallback(Callback):
    """
    Real-time 3D visualization callback for training monitoring.

    Provides interactive visualization of the flock simulation during training,
    allowing researchers to observe emergent behaviors, thermal dynamics, and
    control policy evolution in real-time. Can also log visualization frames
    to WandB for later review.

    The callback manages the PyVista rendering window lifecycle and ensures
    proper resource cleanup. It only activates in interactive mode to avoid
    blocking headless training runs.
    """

    def __init__(
        self,
        auto_close       : bool              = True,
        log_to_wandb     : bool              = True,
        start_epoch      : int               = 0,
        update_frequency : int               = 10,
        visualizer       : Visualizer | None = None,
        watch_run        : bool              = False
    ):
        """
        Configure visualization parameters for training integration.

        Args:
            auto_close       : Automatically close window when training ends
            log_to_wandb     : Log visualization frames to WandB (controlled by vista.log_video)
            start_epoch      : Epoch to start visualization (0 = immediate)
            update_frequency : Update visualization every N batches
            visualizer       : Pre-configured Visualizer instance for rendering
            watch_run        : Show live visualization window (from --watch flag)
        """
        super().__init__()
        self.auto_close       = auto_close
        self.log_to_wandb     = log_to_wandb
        self.start_epoch      = start_epoch
        self.update_frequency = update_frequency
        self.visualizer       = visualizer
        self.watch_run        = watch_run

        self.batch_counter        = 0
        self.frames_buffer        = []
        self.visualization_active = False
        self.video_plotter        = None  # Separate plotter for video capture
        self.live_plotter         = None  # Separate plotter for live window

    def on_fit_start(
        self,
        trainer   : Trainer,
        pl_module : LightningModule
    ):
        """
        Initialize visualization when training begins.

        Sets up the rendering context and prepares the visualizer for
        receiving batch data. Can set up video logging (vista.log_video)
        and/or live window display (--watch flag) independently.
        """
        if not self.visualizer:
            return

        if trainer.current_epoch >= self.start_epoch:
            self.visualization_active = True
            
            # Check if we should log videos to WandB
            should_log_video = (
                self.log_to_wandb and 
                trainer.logger and 
                hasattr(self.visualizer, 'vista') and 
                self.visualizer.vista.log_video
            )
            
            if should_log_video:
                print("📹 Visualization frames will be logged to WandB")
                # Setup video capture plotter if needed
                from pyvista import Plotter
                self.video_plotter = Plotter(off_screen=True, window_size=self.visualizer.vista.window_size)
            
            if self.watch_run:
                print("🎨 Live visualization window opening...")
                # The main visualizer plotter is used for live display
                self.live_plotter = self.visualizer.plotter

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
        threshold is met. Handles both video capture (if log_video=True)
        and live window updates (if --watch=True) independently.
        """
        if not self.visualization_active or not self.visualizer:
            return

        self.batch_counter += 1
        if self.batch_counter % self.update_frequency != 0:
            return

        try:
            # Update the visualizer state with new batch data
            self.visualizer.update(batch)
            
            # Handle live window display if --watch is enabled
            if self.watch_run and self.live_plotter:
                self.visualizer.render()
            
            # Handle video capture for WandB if log_video is enabled
            if self.video_plotter and self.log_to_wandb and trainer.logger:
                # Render to the off-screen plotter for video capture
                # Note: This requires duplicating the rendering logic
                # In a production system, we'd refactor Visualizer to support multiple plotters
                self.video_plotter.clear()
                # Copy visualization state to video plotter
                # For now, just capture from the main plotter if available
                if self.visualizer.plotter:
                    frame = self.visualizer.plotter.screenshot(return_img=True)
                    self.frames_buffer.append(frame)
                    
                    # Log video to WandB every N batches
                    if len(self.frames_buffer) >= 30:  # About 1 second at 30fps
                        self._log_video_to_wandb(trainer, pl_module)
                
        except Exception as e:
            print(f"⚠️ Visualization update failed: {e}")
            self.visualization_active = False

    def _log_video_to_wandb(
        self,
        trainer   : Trainer,
        pl_module : LightningModule
    ):
        """Log accumulated frames as video to WandB."""
        if not self.frames_buffer or not trainer.logger:
            return
            
        try:
            import wandb
            import numpy as np
            
            # Convert frames to numpy array
            video_array = np.array(self.frames_buffer)
            
            # Log to WandB
            if hasattr(trainer.logger, 'experiment'):
                trainer.logger.experiment.log({
                    "visualization/policy_behavior": wandb.Video(
                        video_array, 
                        fps=30,
                        format="mp4"
                    ),
                    "global_step": trainer.global_step
                })
            elif hasattr(trainer.logger, 'log_video'):
                # Alternative method if available
                trainer.logger.log_video(
                    "visualization/policy_behavior",
                    [video_array],
                    fps=30
                )
            else:
                # Fallback: Use wandb directly if available
                wandb.log({
                    "visualization/policy_behavior": wandb.Video(
                        video_array, 
                        fps=30,
                        format="mp4"
                    ),
                    "global_step": trainer.global_step
                })
            
            # Clear buffer
            self.frames_buffer.clear()
            
        except Exception as e:
            print(f"⚠️ Failed to log video to WandB: {e}")

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
        """
        if not self.visualization_active or not self.visualizer:
            return

        if batch_idx == 0:
            try:
                self.visualizer.update(batch)
                self.visualizer.render()
            except Exception as e:
                print(f"⚠️ Validation visualization failed: {e}")

    def on_train_epoch_end(
        self,
        trainer   : Trainer,
        pl_module : LightningModule
    ):
        """
        Log any remaining frames at epoch end.

        Ensures all captured frames are sent to WandB before moving
        to the next epoch.
        """
        if self.log_to_wandb and self.frames_buffer:
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
        """
        if not self.visualizer or self.visualization_active:
            return

        if trainer.current_epoch >= self.start_epoch:
            self.visualization_active = True
            print(f"🎨 Starting visualization at epoch {trainer.current_epoch}")

    def on_fit_end(
        self,
        trainer   : Trainer,
        pl_module : LightningModule
    ):
        """
        Clean up visualization resources when training completes.

        Logs any remaining frames to WandB and closes any open windows.
        Ensures proper cleanup even if training ends early.
        """
        if not self.visualizer:
            return

        # Log any remaining frames
        if self.log_to_wandb and self.frames_buffer:
            self._log_video_to_wandb(trainer, pl_module)

        # Close video plotter if it exists
        if self.video_plotter:
            try:
                self.video_plotter.close()
            except Exception:
                pass
            finally:
                self.video_plotter = None

        # Close live visualization window if requested
        if self.visualization_active and self.auto_close and self.watch_run:
            try:
                self.visualizer.close()
                print("🎨 Live visualization window closed")
            except Exception as e:
                print(f"⚠️ Failed to close visualization: {e}")
            finally:
                self.visualization_active = False

    def on_exception(
        self,
        trainer   : Trainer,
        pl_module : LightningModule,
        exception : BaseException
    ):
        """
        Ensure proper cleanup when training fails.

        Closes visualization resources even when training is
        interrupted by an exception, preventing resource leaks.
        """
        if self.visualizer and self.visualization_active:
            try:
                self.visualizer.close()
            except:
                pass
            finally:
                self.visualization_active = False
