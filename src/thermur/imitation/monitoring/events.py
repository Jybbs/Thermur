"""
Event logging system for tracking agent-level decisions and state transitions.

This module provides detailed logging capabilities for debugging swarm behavior,
tracking individual agent decisions, CBF activations, and critical events during
training and simulation. It integrates with PyTorch Lightning's logging system
and provides structured outputs for post-hoc analysis.
"""
from collections                         import Counter, defaultdict
from config.imitation.schemas.monitoring import MonitoringModel
from pytorch_lightning                   import LightningModule
from tensordict                          import TensorDict
from time                                import perf_counter
from torch                               import where

import wandb


class EventLogger:
    """
    Unified event logger that tracks both rates and detailed event data.
    
    Logs event rates as standard metrics for monitoring trends, while
    sampling detailed event data to W&B tables for debugging.
    """
    
    
    def __init__(
        self,
        activation_tolerance : float,
        max_temperature      : float,
        monitoring           : MonitoringModel
    ):
        """
        Initialize the event logger.
        
        Args:
            activation_tolerance : CBF activation tolerance from safety config
            max_temperature      : Maximum safe temperature from flock config
            monitoring           : Monitoring configuration model
        """
        self.cbf_tolerance   = activation_tolerance
        self.cbf_threshold   = max_temperature - activation_tolerance
        self.event_types     = monitoring.event_types
        self.max_temperature = max_temperature
        self.prefix          = monitoring.prefix
        self.sample_every    = monitoring.event_sample_every
        
        self.event_buffer = defaultdict(list)
        self.event_counts = Counter()
        self.event_data   = defaultdict(list) 
        self.start_time   = perf_counter()
        self.total_steps  = 0
    
    def _flush_events_to_table(
        self,
        event_type : str,
        module     : LightningModule
    ):
        """
        Push buffered events to W&B table.
        
        Args:
            event_type : Type of event to flush
            module     : Lightning module with logger
        """
        if not self.event_data[event_type]:
            return
            
        event_config = self.event_types[event_type]
        columns      = ["step"] + event_config["columns"]
        
        data = [
            [event_dict["step"]] + [
                event_dict.get(col) for col in event_config["columns"]
            ]
            for event_dict in self.event_data[event_type]
        ]
        
        table = wandb.Table(columns, data)
        module.logger.experiment.log({
            f"{self.prefix}{event_type}_details": table
        })
        
        self.event_data[event_type].clear()
    
    def _log_event(
        self, 
        event_type : str,
        module     : LightningModule,
        **event_data
    ):
        """
        Log both rate metrics and sampled event details.
        
        Args:
            event_type   : Type of event from EVENT_TYPES
            module       : Lightning module for logging
            **event_data : Event-specific data fields
        """
        self.event_counts[event_type] += 1
        
        module.log(
            name     = self.prefix + self.event_types[event_type]["rate_metric"],
            value    = self.event_counts[event_type] / max(self.total_steps, 1),
            on_epoch = True,
            on_step  = True
        )
        
        event_dict = {"step": module.global_step}
        event_dict.update(event_data)
        self.event_data[event_type].append(event_dict)
        
        if (
            module.global_step % self.sample_every == 0
                and module.logger 
                and hasattr(module.logger, 'experiment')
        ):
            self._flush_events_to_table(event_type, module)
    
    def analyze_batch(
        self, 
        batch  : TensorDict, 
        module : LightningModule
    ) -> dict[str, int]:
        """
        Scan batch for all event types and log them.
        
        Args:
            batch  : TensorDict containing simulation state
            module : Lightning module for logging
            
        Returns:
            Dictionary with counts of each event type detected
        """
        if "temperature" not in batch:
            return {}
            
        self.total_steps += batch["temperature"].shape[0]
       
        analysis = Counter()
        temps    = batch["temperature"].flatten()
        
        if (violation_ids := where(temps > self.max_temperature)[0]).numel():
            for idx in violation_ids.tolist():
                self._log_event(
                    agent_id    = idx,
                    event_type  = "thermal_violation",
                    excess      = temps[idx].item() - self.max_temperature,
                    module      = module,
                    position    = batch["position"][idx].cpu().tolist(),
                    temperature = temps[idx].item()
                )
            analysis["thermal_violations"] = violation_ids.numel()
        
        if (near_miss_ids := where(
            (temps > self.cbf_threshold) & (temps <= self.max_temperature)
        )[0]).numel():
            for idx in near_miss_ids.tolist():
                self._log_event(
                    agent_id    = idx,
                    event_type  = "near_miss",
                    margin      = self.max_temperature - temps[idx].item(),
                    module      = module,
                    position    = batch["position"][idx].cpu().tolist(),
                    temperature = temps[idx].item()
                )
            analysis["near_misses"] = near_miss_ids.numel()
        
        required = {"cbf_active", "u_nominal", "u_safe"}
        if (required <= batch.keys() and
            (active_ids := where(batch["cbf_active"])[0]).numel()):
            
            for agent_id in active_ids.tolist():
                control_diff = (
                    batch["u_safe"][agent_id] - batch["u_nominal"][agent_id]
                ).norm().item()
                safety_margin = (
                    self.max_temperature - batch["temperature"][agent_id].item()
                )

                self._log_event(
                    agent_id      = agent_id,
                    control_diff  = control_diff,
                    event_type    = "cbf_activation",
                    module        = module,
                    safety_margin = safety_margin,
                    temperature   = batch["temperature"][agent_id].item()
                )

            analysis["cbf_activations"] = active_ids.numel()
        
        return dict(analysis)
    
    def flush_all(self, module: LightningModule):
        """
        Flush all remaining events to tables.
        
        Args:
            module : Lightning module with logger
        """
        for event_type in self.event_types:
            if self.event_data[event_type]:
                self._flush_events_to_table(event_type, module)
    
    def get_event_summary(self) -> dict:
        """
        Get summary statistics of logged events.
        
        Returns:
            Dictionary containing event counts and timing information
        """
        return {
            "elapsed_time"   : perf_counter() - self.start_time,
            "events_by_type" : dict(self.event_counts),
            "total_events"   : sum(self.event_counts.values()),
            "total_steps"    : self.total_steps
        }
    
    def reset_epoch_metrics(self):
        """
        Reset counters at epoch boundaries.
        """
        self.event_counts.clear()
        self.total_steps = 0