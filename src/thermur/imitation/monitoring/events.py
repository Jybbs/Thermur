"""
Event logging system for tracking agent-level decisions and state transitions.

This module provides detailed logging capabilities for debugging swarm behavior,
tracking individual agent decisions, CBF activations, and critical events during
training and simulation. It integrates with PyTorch Lightning's logging system
and provides structured outputs for post-hoc analysis.
"""
from __future__   import annotations
from collections  import Counter, defaultdict
from config.types import EventConfig
from time         import perf_counter
from typing       import Any, TYPE_CHECKING
from wandb        import Table

if TYPE_CHECKING:
    from config.imitation.controller import ThresholdsModel
    from config.imitation.monitoring import EventsModel
    from pytorch_lightning           import LightningModule
    from tensordict                  import TensorDictBase
    from torch                       import Tensor


class EventLogger:
    """
    Unified event logger that tracks both rates and detailed event data.

    Logs event rates as standard metrics for monitoring trends, while
    sampling detailed event data to W&B tables for debugging.
    """


    def __init__(
        self,
        events     : EventsModel,
        thresholds : ThresholdsModel
    ):
        """
        Initialize the event logger.

        Args:
            events     : Event logging configuration model
            thresholds : Safety threshold configuration from controller domain
        """
        self.cbf_tolerance   = thresholds.activation_tolerance
        self.cbf_threshold   = thresholds.max_temperature - thresholds.activation_tolerance
        self.max_temperature = thresholds.max_temperature
        self.sample_every    = events.event_sample_every
        self.start_time      = perf_counter()
        self.total_steps     = 0

        self.event_buffer : defaultdict[str, list[Any]]            = defaultdict(list)
        self.event_counts : Counter[str]                           = Counter()
        self.event_data   : defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    def _batch_log_masked_events(
        self,
        batch      : TensorDictBase,
        event_type : str,
        mask       : Tensor,
        module     : LightningModule,
        temps      : Tensor,
        **fields   : Tensor
    ) -> int:
        """
        Log events for all agents matching a boolean mask.

        Processes a batch of agents identified by a boolean mask, extracting their
        data and logging individual events. Supports arbitrary additional fields
        passed as keyword arguments, which will be masked and logged per agent.

        Args:
            batch      : TensorDict containing full batch state including positions
            event_type : Name of the event type being logged (e.g., "thermal_violation")
            mask       : Boolean tensor indicating which agents to log events for
            module     : Lightning module providing access to the logger
            temps      : Temperature tensor aligned with mask dimensions
            **fields   : Additional scalar fields to log, as tensors matching batch size

        Returns:
            Number of events successfully logged
        """
        count = int(mask.sum().item())
        if not count:
            return 0

        indices          = mask.nonzero(as_tuple=False).squeeze(-1)
        masked_temps     = temps[mask]
        masked_positions = batch["position"][mask]

        for i in range(count):
            event_data: dict[str, Any] = {
                "agent_id"    : int(indices[i].item()),
                "position"    : masked_positions[i].cpu().tolist(),
                "temperature" : float(masked_temps[i].item())
            }

            for field_name, field_tensor in fields.items():
                event_data[field_name] = field_tensor[mask][i].item()

            self._log_event(
                event_type = event_type,
                module     = module,
                **event_data
            )

        return count

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

        event_config = self._get_event_types()[event_type]
        columns      = ["step"] + event_config["columns"]

        data = [
            [event_dict["step"]] + [
                event_dict.get(col) for col in event_config["columns"]
            ]
            for event_dict in self.event_data[event_type]
        ]

        table = Table(columns, data)
        if experiment := getattr(module.logger, 'experiment', None):
            experiment.log({f"events/{event_type}_details": table})

        self.event_data[event_type].clear()

    def _get_event_types(self) -> dict[str, EventConfig]:
        """
        Returns the configuration for each event type.

        Returns:
            Dictionary mapping event type names to their configurations
        """
        return {
            "cbf_activation": EventConfig(
                columns = ["agent_id", "temperature", "safety_margin", "control_diff"],
                rate    = "cbf_activation_rate"
            ),
            "near_miss": EventConfig(
                columns = ["agent_id", "temperature", "position", "margin"],
                rate    = "near_miss_rate"
            ),
            "thermal_violation": EventConfig(
                columns = ["agent_id", "temperature", "position", "excess"],
                rate    = "thermal_violation_rate"
            )
        }

    def _log_event(
        self,
        event_type : str,
        module     : LightningModule,
        **event_data: Any
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
            name     = "events/" + self._get_event_types()[event_type]["rate"],
            value    = self.event_counts[event_type] / max(self.total_steps, 1),
            on_epoch = True,
            on_step  = True
        )

        event_dict = {"step": module.global_step, **event_data}
        self.event_data[event_type].append(event_dict)

        if (
            module.global_step % self.sample_every == 0
                and module.logger
                and hasattr(module.logger, 'experiment')
        ):
            self._flush_events_to_table(event_type, module)

    def analyze_batch(
        self,
        batch  : TensorDictBase,
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

        analysis: Counter[str] = Counter()
        temps = batch["temperature"].flatten()

        violation_mask = temps > self.max_temperature
        count = self._batch_log_masked_events(
            batch      = batch,
            event_type = "thermal_violation",
            excess     = temps - self.max_temperature,
            mask       = violation_mask,
            module     = module,
            temps      = temps
        )
        if count:
            analysis["thermal_violations"] = count

        near_miss_mask = (temps > self.cbf_threshold) & (temps <= self.max_temperature)
        count = self._batch_log_masked_events(
            batch      = batch,
            event_type = "near_miss",
            margin     = self.max_temperature - temps,
            mask       = near_miss_mask,
            module     = module,
            temps      = temps
        )
        if count:
            analysis["near_misses"] = count

        required = {"cbf_active", "u_nominal", "u_safe"}
        if all(key in batch for key in required):
            cbf_mask = batch["cbf_active"].bool()
            count = self._batch_log_masked_events(
                batch          = batch,
                control_diff   = (batch["u_safe"] - batch["u_nominal"]).norm(dim=-1),
                event_type     = "cbf_activation",
                mask           = cbf_mask,
                module         = module,
                safety_margin  = self.max_temperature - batch["temperature"],
                temps          = batch["temperature"]
            )
            if count:
                analysis["cbf_activations"] = count

        return dict(analysis)

    def flush_all(self, module: LightningModule):
        """
        Flush all remaining events to tables.

        Args:
            module  Lightning module with logger
        """
        for event_type in self._get_event_types():
            if self.event_data[event_type]:
                self._flush_events_to_table(event_type, module)

    def get_event_summary(self) -> dict[str, float | int]:
        """
        Get summary statistics of logged events.

        Returns:
            Dictionary containing event counts and timing information
        """
        summary: dict[str, float | int] = {
            "elapsed_time" : perf_counter() - self.start_time,
            "total_events" : sum(self.event_counts.values()),
            "total_steps"  : self.total_steps
        }

        for event_type, count in self.event_counts.items():
            summary[f"events/{event_type}"] = count

        return summary

    def reset_epoch_metrics(self):
        """
        Reset counters at epoch boundaries.
        """
        self.event_counts.clear()
        self.total_steps = 0
