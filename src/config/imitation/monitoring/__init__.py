"""
Monitoring configuration domain.

Provides schemas and stores for monitoring components including
metrics collection, event tracking, and performance analysis.
"""
# Re-export schemas for clean imports
from .schemas import (
    EventsModel,
    MetricsModel
)

__all__ = [
    # Schemas
    "EventsModel",
    "MetricsModel"
]