"""
Monitoring and evaluation infrastructure for imitation learning.

This module provides comprehensive monitoring capabilities for training and
evaluation, including metric collection, event logging, and debugging utilities.
The monitoring system tracks both aggregate metrics and individual agent behaviors
to provide insights into swarm performance and emergent phenomena.
"""
from .events  import *
from .metrics import *