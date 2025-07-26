"""Monitoring configuration schemas.

This subpackage contains configuration models for metrics collection and
event logging during training and evaluation:

- monitoring.py: Unified monitoring configuration for metrics and events

The monitoring configuration enables comprehensive tracking of training
progress, system performance, and behavioral metrics. It supports both
real-time visualization and persistent logging for post-hoc analysis.
Metrics include cohesion, separation, thermal safety, and computational
performance indicators.
"""
from .events  import *
from .metrics import *