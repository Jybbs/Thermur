"""
Event logging configuration schemas.

This module defines configuration for event detection and logging during
training and evaluation, including CBF activations, thermal violations,
near misses, and topology changes.
"""
from pydantic import BaseModel, Field, PositiveInt
from typing   import Any


class EventsModel(BaseModel, extra="forbid"):
    """
    Configuration for event detection and logging system.
    
    Defines parameters for detecting and logging critical events during
    simulation, including safety violations and control interventions.
    """
    event_sample_every: PositiveInt = Field(
        default     = 100,
        description = (
            "Global steps between detailed event sampling for logging to tables, "
            "controlling data volume and analysis granularity."
        )
    )
    event_types: dict[str, dict[str, Any]] = Field(
        default     = {
            "cbf_activation": {
                "columns"     : ["agent_id", "temperature", "control_diff", "safety_margin"],
                "name"        : "CBF Activations",
                "rate_metric" : "cbf_activation_rate"
            },
            "near_miss": {
                "columns"     : ["agent_id", "temperature", "margin", "position"],
                "name"        : "Near Misses",
                "rate_metric" : "near_miss_rate"
            },
            "thermal_violation": {
                "columns"     : ["agent_id", "temperature", "excess", "position"],
                "name"        : "Thermal Violations",
                "rate_metric" : "thermal_violation_rate"
            },
            "topology_change": {
                "columns"     : ["agent_id", "neighbors_added", "neighbors_lost", "neighbor_count"],
                "name"        : "Topology Changes",
                "rate_metric" : "topology_change_rate"
            }
        },
        description = (
            "Event type definitions mapping identifiers to configurations including "
            "data columns, display names, and rate tracking metric names."
        )
    )
    prefix: str = Field(
        default     = "events/",
        description = (
            "Namespace prefix for event rate metrics in logging systems like "
            "Weights & Biases for organized metric hierarchies."
        )
    )