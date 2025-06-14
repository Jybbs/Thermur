"""
Core components for the Thermur project.

This package provides the fundamental building blocks including the environment,
orchestration, data structures, and geometric utilities.
"""
from thermur.core.cli          import cli_main
from thermur.core.env          import ThermurEnv
from thermur.core.geometry     import compute_edge_index
from thermur.core.orchestrator import ImitationLoss, TrainingOrchestrator
from thermur.core.structures   import SwarmData, SwarmDataSpec

__all__ = [
    "cli_main",
    "compute_edge_index",
    "ImitationLoss",
    "SwarmData",
    "SwarmDataSpec",
    "ThermurEnv",
    "TrainingOrchestrator",
]
