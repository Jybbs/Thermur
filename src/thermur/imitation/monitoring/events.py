"""
Event logging system for tracking agent-level decisions and state transitions.

This module provides detailed logging capabilities for debugging swarm behavior,
tracking individual agent decisions, CBF activations, and critical events during
training and simulation. It integrates with PyTorch Lightning's logging system
and provides structured outputs for post-hoc analysis.
"""
from collections       import Counter, defaultdict
from dataclasses       import dataclass
from pytorch_lightning import LightningModule
from tensordict        import TensorDict
from time              import perf_counter
from torch             import where
from typing            import Any

import wandb


@dataclass
class Event:
    """
    Single event instance with metadata.
    """
    data       : dict[str, Any]
    event_type : str
    step       : int
    
    def to_row(self, columns: list[str]) -> list[Any]:
        """
        Convert event to table row format.
        
        Args:
            columns : Column names to extract from data
            
        Returns:
            List of values for W&B table row
        """
        return [self.step] + [self.data.get(col) for col in columns]


@dataclass
class EventType:
    """
    Configuration for a specific type of event.
    """
    columns     : list[str]
    name        : str
    rate_metric : str


class EventLogger:
    """
    Unified event logger that tracks both rates and detailed event data.
    
    Logs event rates as standard metrics for monitoring trends, while
    sampling detailed event data to W&B tables for debugging.
    """
    
    EVENT_TYPES = {
        "cbf_activation": EventType(
            columns     = ["agent_id", "temperature", "control_diff", "safety_margin"],
            name        = "CBF Activations",
            rate_metric = "events/cbf_activation_rate"
        ),
        "near_miss": EventType(
            columns     = ["agent_id", "temperature", "margin", "position"],
            name        = "Near Misses",
            rate_metric = "events/near_miss_rate"
        ),
        "thermal_violation": EventType(
            columns     = ["agent_id", "temperature", "excess", "position"],
            name        = "Thermal Violations",
            rate_metric = "events/thermal_violation_rate"
        ),
        "topology_change": EventType(
            columns     = [
                "agent_id", "neighbors_added", "neighbors_lost", "neighbor_count"
            ],
            name        = "Topology Changes",
            rate_metric = "events/topology_change_rate"
        )
    }
    
    def __init__(
        self,
        cbf_tolerance   : float = 3.0,
        max_temperature : float = 475.0,
        sample_every    : int   = 100
    ):
        """
        Initialize the event logger.
        
        Args:
            cbf_tolerance   : Temperature tolerance for CBF activation
            max_temperature : Maximum safe temperature for thermal events
            sample_every    : Steps between detailed event sampling
        """
        self.cbf_tolerance   = cbf_tolerance
        self.cbf_threshold   = max_temperature - cbf_tolerance
        self.max_temperature = max_temperature
        self.sample_every    = sample_every
        
        self.event_buffer = defaultdict(list)
        self.event_counts = Counter()
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
        if not self.event_buffer[event_type]:
            return
            
        event_config = self.EVENT_TYPES[event_type]
        columns = ["step"] + event_config.columns
        
        data = [
            event.to_row(event_config.columns) 
            for event in self.event_buffer[event_type]
        ]
        
        table = wandb.Table(columns=columns, data=data)
        module.logger.experiment.log({
            f"events/{event_type}_details": table
        })
        
        self.event_buffer[event_type].clear()
    
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
            name     = self.EVENT_TYPES[event_type].rate_metric,
            on_epoch = True,
            on_step  = True,
            prog_bar = False,
            value    = self.event_counts[event_type] / max(self.total_steps, 1)
        )
        
        event = Event(
            data       = event_data,
            event_type = event_type,
            step       = module.global_step
        )
        self.event_buffer[event_type].append(event)
        
        if (
            module.global_step % self.sample_every == 0
                and module.logger 
                and hasattr(module.logger, 'experiment')
        ):
            self._flush_events_to_table(event_type, module)
    
    def analyze_batch(self, batch: TensorDict, module: LightningModule) -> dict:
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
            for i in range(violation_ids.shape[0]):

                idx = violation_ids[i].item()
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
            [
                self._log_event(
                    agent_id    = idx,
                    event_type  = "near_miss",
                    margin      = self.max_temperature - temps[idx].item(),
                    module      = module,
                    position    = batch["position"][idx].cpu().tolist(),
                    temperature = temps[idx].item()
                )
                for idx in near_miss_ids.tolist()
            ]
            analysis["near_misses"] = near_miss_ids.numel()
        
        required = {"cbf_active", "u_nominal", "u_safe"}
        if (required <= batch.keys() and
            (active_ids := where(batch["cbf_active"])[0]).numel()):
            
            for i, agent_id in enumerate(active_ids.tolist()):
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
        for event_type in self.EVENT_TYPES:
            if self.event_buffer[event_type]:
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