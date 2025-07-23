"""
Event logging system for tracking agent-level decisions and state transitions.

This module provides detailed logging capabilities for debugging swarm behavior,
tracking individual agent decisions, CBF activations, and critical events during
training and simulation. It integrates with PyTorch Lightning's logging system
and provides structured outputs for post-hoc analysis.
"""
from dataclasses import dataclass
from pathlib     import Path
from tensordict  import TensorDict
from torch       import Tensor
from typing      import Optional

import json
import time


@dataclass
class AgentEvent:
    """
    Encapsulates all information about a discrete event that occurred
    to a specific agent during simulation or training.
    """
    agent_id   : int
    data       : dict
    event_type : str
    timestamp  : float
    
    def to_dict(self) -> dict:
        """
        Convert event to dictionary for serialization.
        
        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "agent_id"   : self.agent_id,
            "data"       : self.data,
            "event_type" : self.event_type,
            "timestamp"  : self.timestamp
        }


class EventLogger:
    """
    Tracks and logs agent-level events for debugging and analysis.
    
    This logger captures detailed information about individual agent behaviors,
    safety system activations, and critical events. It provides structured
    logging that can be analyzed offline to understand emergent behaviors
    and debug issues in the swarm.
    """
    
    def __init__(
        self, 
        buffer_size     : int = 1000,
        cbf_tolerance   : float = 3.0,
        enable_file_log : bool = True,
        log_dir         : Optional[Path] = None,
        max_temperature : float = 475.0
    ):
        """
        Sets up event buffering, file logging, and temperature thresholds
        for detecting critical events.
        
        Args:
            buffer_size     : Number of events to buffer before writing
            cbf_tolerance   : Temperature tolerance for CBF activation
            enable_file_log : Whether to write events to file
            log_dir         : Directory for saving event logs
            max_temperature : Maximum safe temperature for thermal events
        """
        self.buffer_size     = buffer_size
        self.cbf_tolerance   = cbf_tolerance
        self.cbf_threshold   = max_temperature - cbf_tolerance
        self.enable_file_log = enable_file_log
        self.log_dir         = log_dir
        self.max_temperature = max_temperature
        
        self.event_buffer = []
        self.event_counts = {}
        self.start_time   = time.time()
        
        if enable_file_log and log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.event_file = self.log_dir / "agent_events.jsonl"
    
    def _add_event(self, event: AgentEvent):
        """
        Maintains event counts by type and flushes buffer when full.
        
        Args:
            event : AgentEvent to log
        """
        self.event_buffer.append(event)
        self.event_counts[event.event_type] = (
            self.event_counts.get(event.event_type, 0) + 1
        )
        
        if len(self.event_buffer) >= self.buffer_size:
            self.flush()
    
    def _get_timestamp(self) -> float:
        """
        Get elapsed time since logger initialization.
        
        Returns:
            Seconds elapsed since logger creation
        """
        return time.time() - self.start_time
    
    def analyze_batch(self, batch: TensorDict) -> dict:
        """
        Scans the batch for thermal violations, CBF activations, and
        near-miss events, logging each occurrence with relevant context.
        
        Args:
            batch : TensorDict containing simulation state
            
        Returns:
            Dictionary with counts of each event type detected
        """
        analysis = {
            "cbf_activations"    : 0,
            "near_misses"        : 0,
            "thermal_violations" : 0
        }
        
        if "temperature" in batch:
            temps = batch["temperature"]
            if temps.dim() > 1:
                temps = temps.squeeze(-1)
                
            violations_mask = temps > self.max_temperature
            if violations_mask.any():
                violation_ids = violations_mask.nonzero(as_tuple=True)[0]
                self.log_thermal_violation(
                    violation_ids,
                    batch["position"][violations_mask],
                    temps[violations_mask]
                )
                analysis["thermal_violations"] = violation_ids.numel()
            
            near_miss_mask = (temps > self.cbf_threshold) & (~violations_mask)
            if near_miss_mask.any():
                near_miss_ids = near_miss_mask.nonzero(as_tuple=True)[0]
                for agent_id in near_miss_ids:
                    self.log_near_miss(
                        int(agent_id.item()),
                        self.max_temperature - temps[agent_id].item(),
                        batch["position"][agent_id],
                        temps[agent_id].item()
                    )
                analysis["near_misses"] = near_miss_ids.numel()
        
        if "cbf_active" in batch and batch["cbf_active"].any():
            active_ids = batch["cbf_active"].nonzero(as_tuple=True)[0]
            if "u_nominal" in batch and "u_safe" in batch:
                self.log_cbf_activation(
                    active_ids,
                    batch["temperature"][active_ids],
                    batch["u_nominal"][active_ids],
                    batch["u_safe"][active_ids]
                )
            analysis["cbf_activations"] = active_ids.numel()
        
        return analysis
    
    def flush(self):
        """
        Writes all buffered events to the JSON Lines file and clears
        the buffer. No-op if file logging is disabled.
        """
        if not self.enable_file_log or not self.event_buffer:
            return
            
        with open(self.event_file, "a") as f:
            for event in self.event_buffer:
                json.dump(event.to_dict(), f)
                f.write("\n")
        
        self.event_buffer.clear()
    
    def get_event_summary(self) -> dict:
        """
        Get summary statistics of logged events.
        
        Returns:
            Dictionary containing event counts and timing information
        """
        total_events = sum(self.event_counts.values())
        return {
            "buffer_size"     : len(self.event_buffer),
            "elapsed_time"    : self._get_timestamp(),
            "events_by_type"  : self.event_counts.copy(),
            "total_events"    : total_events
        }
    
    def log_cbf_activation(
        self,
        agent_ids    : Tensor,
        temperatures : Tensor,
        u_nominal    : Tensor,
        u_safe       : Tensor,
        safety_margins : Optional[Tensor] = None
    ):
        """
        Records when the CBF safety filter modifies control commands to
        maintain thermal safety constraints.
        
        Args:
            agent_ids      : IDs of agents where CBF activated [N_active]
            temperatures   : Current temperatures [N_active]
            u_nominal      : Nominal control before CBF [N_active, 3]
            u_safe         : Safe control after CBF [N_active, 3]
            safety_margins : Optional safety margin values [N_active]
        """
        timestamp = self._get_timestamp()
        
        for i in range(agent_ids.shape[0]):
            control_diff = (u_safe[i] - u_nominal[i]).norm().item()
            
            event = AgentEvent(
                agent_id   = int(agent_ids[i].item()),
                data       = {
                    "control_diff"   : float(control_diff),
                    "safety_margin"  : (float(safety_margins[i].item()) 
                                       if safety_margins is not None else None),
                    "temperature"    : float(temperatures[i].item()),
                    "u_nominal_norm" : float(u_nominal[i].norm().item()),
                    "u_safe_norm"    : float(u_safe[i].norm().item())
                },
                event_type = "cbf_activation",
                timestamp  = timestamp
            )
            self._add_event(event)
    
    def log_communication_change(
        self,
        agent_id      : int,
        new_neighbors : list[int],
        old_neighbors : list[int],
        position      : Tensor
    ):
        """
        Records when agents gain or lose communication links due to
        relative motion affecting the proximity-based network.
        
        Args:
            agent_id      : ID of the agent
            new_neighbors : New neighbor IDs
            old_neighbors : Previous neighbor IDs
            position      : Current position of the agent
        """
        added   = set(new_neighbors) - set(old_neighbors)
        removed = set(old_neighbors) - set(new_neighbors)
        
        if added or removed:
            event = AgentEvent(
                agent_id   = agent_id,
                data       = {
                    "neighbor_count"  : len(new_neighbors),
                    "neighbors_added" : list(added),
                    "neighbors_lost"  : list(removed),
                    "position"        : position.cpu().tolist()
                },
                event_type = "topology_change",
                timestamp  = self._get_timestamp()
            )
            self._add_event(event)
    
    def log_near_miss(
        self,
        agent_id    : int,
        margin      : float,
        position    : Tensor,
        temperature : float
    ):
        """
        Records when agents enter the CBF activation zone but haven't
        yet violated safety constraints.
        
        Args:
            agent_id    : ID of the agent
            margin      : Temperature margin to safety threshold
            position    : Current position
            temperature : Current temperature
        """
        event = AgentEvent(
            agent_id   = agent_id,
            data       = {
                "margin"      : float(margin),
                "position"    : position.cpu().tolist(),
                "temperature" : float(temperature),
                "threshold"   : self.cbf_threshold
            },
            event_type = "near_miss",
            timestamp  = self._get_timestamp()
        )
        self._add_event(event)
    
    def log_thermal_violation(
        self, 
        agent_ids    : Tensor,
        positions    : Tensor,
        temperatures : Tensor
    ):
        """
        Records critical failures where agents exceeded the maximum safe
        temperature despite safety measures.
        
        Args:
            agent_ids    : IDs of agents with violations [N_violations]
            positions    : Positions of violating agents [N_violations, 3]
            temperatures : Temperatures of violating agents [N_violations]
        """
        timestamp = self._get_timestamp()
        
        for i in range(agent_ids.shape[0]):
            event = AgentEvent(
                agent_id   = int(agent_ids[i].item()),
                data       = {
                    "excess"      : float(temperatures[i].item() - self.max_temperature),
                    "position"    : positions[i].cpu().tolist(),
                    "temperature" : float(temperatures[i].item())
                },
                event_type = "thermal_violation",
                timestamp  = timestamp
            )
            self._add_event(event)
    
    def log_trajectory_anomaly(
        self,
        acceleration : Tensor,
        agent_id     : int,
        jerk_norm    : float,
        position     : Tensor,
        velocity     : Tensor
    ):
        """
        Records unusual motion patterns that may indicate control issues
        or numerical instabilities.
        
        Args:
            acceleration : Current acceleration
            agent_id     : ID of the agent
            jerk_norm    : Magnitude of jerk (rate of acceleration change)
            position     : Current position
            velocity     : Current velocity
        """
        event = AgentEvent(
            agent_id   = agent_id,
            data       = {
                "accel_norm"    : float(acceleration.norm().item()),
                "jerk_norm"     : float(jerk_norm),
                "position"      : position.cpu().tolist(),
                "velocity_norm" : float(velocity.norm().item())
            },
            event_type = "trajectory_anomaly",
            timestamp  = self._get_timestamp()
        )
        self._add_event(event)