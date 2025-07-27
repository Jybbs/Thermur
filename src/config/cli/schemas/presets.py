"""
Training preset configurations with hyperparameter overrides.

This module defines preset configurations that provide pre-tuned hyperparameter
sets for different training scenarios. Each preset includes overrides for key
parameters across the training pipeline.
"""
from pydantic import BaseModel, Field
from typing   import Any


class PresetOverrides(BaseModel, extra="forbid"):
    """
    Hyperparameter overrides for a specific preset.
    
    Uses a dictionary to store Hydra-style override paths and values that will
    be applied when the preset is selected.
    """
    overrides : dict[str, Any] = Field(
        default_factory = dict,
        description     = "Dictionary mapping Hydra paths to override values."
    )


class PresetInfo(BaseModel, extra="forbid"):
    """
    Metadata and configuration for a single training preset.
    """
    best_for  : str              = Field(description = "Use case where this preset excels.")
    desc      : str              = Field(description = "Short description of the preset's purpose.")
    emoji     : str              = Field(description = "Visual indicator for the preset.")
    name      : str              = Field(description = "Preset identifier used in CLI commands.")
    overrides : PresetOverrides  = Field(description = "Hyperparameter overrides for this preset.")


class PresetsModel(BaseModel, extra="forbid"):
    """
    Complete preset system with hyperparameter configurations.
    
    This model defines all available training presets, each with its own
    set of hyperparameter overrides optimized for specific use cases.
    """
    quick : PresetInfo = Field(
        default = PresetInfo(
            best_for  = "Quick experiments & debugging",
            desc      = "Minimal setup for rapid testing",
            emoji     = "⚡",
            name      = "quick",
            overrides = PresetOverrides(overrides={
                "checkpoint.every_n_train_steps"    : 1000,
                "experience.batch_size"             : 32,
                "experience.total_frames"           : 10_000,
                "flock.agent_count"                 : 5,
                "hardware.compile_model"            : False,
                "hardware.precision"                : "16-mixed",
                "metrics.enable_model_summary"      : False,
                "metrics.enable_progress_bar"       : True,
                "metrics.log_every_n_steps"         : 100,
                "metrics.profiler"                  : None,
                "optimizer.early_stopping_patience" : 5,
                "optimizer.gradient_clip_val"       : 1.0,
                "optimizer.learning_rate"           : 1e-3,
                "wandb.mode"                        : "disabled"
            })
        ),
        description = "Quick iteration preset for rapid experimentation."
    )
    standard : PresetInfo = Field(
        default = PresetInfo(
            best_for  = "Regular training runs",
            desc      = "Balanced configuration for most tasks",
            emoji     = "🔥",
            name      = "standard",
            overrides = PresetOverrides(overrides={
                "checkpoint.every_n_train_steps"    : 500,
                "experience.batch_size"             : 64,
                "experience.total_frames"           : 100_000,
                "flock.agent_count"                 : 10,
                "hardware.compile_model"            : True,
                "hardware.precision"                : "16-mixed",
                "metrics.enable_model_summary"      : True,
                "metrics.enable_progress_bar"       : True,
                "metrics.log_every_n_steps"         : 50,
                "metrics.profiler"                  : None,
                "optimizer.early_stopping_patience" : 15,
                "optimizer.gradient_clip_val"       : 0.5,
                "optimizer.learning_rate"           : 3e-4,
                "wandb.mode"                        : "online"
            })
        ),
        description = "Standard training preset with balanced settings."
    )
    large : PresetInfo = Field(
        default = PresetInfo(
            best_for  = "Production & final models",
            desc      = "High-capacity models & longer training",
            emoji     = "💪",
            name      = "large",
            overrides = PresetOverrides(overrides={
                "checkpoint.every_n_train_steps"    : 200,
                "experience.batch_size"             : 128,
                "experience.total_frames"           : 1_000_000,
                "flock.agent_count"                 : 20,
                "hardware.compile_model"            : True,
                "hardware.precision"                : "32-true",
                "metrics.enable_model_summary"      : True,
                "metrics.enable_progress_bar"       : False,
                "metrics.log_every_n_steps"         : 20,
                "metrics.profiler"                  : None,
                "optimizer.early_stopping_patience" : 30,
                "optimizer.gradient_clip_val"       : 0.1,
                "optimizer.learning_rate"           : 1e-4,
                "wandb.mode"                        : "online"
            })
        ),
        description = "Production preset for high-quality models."
    )
    debug : PresetInfo = Field(
        default = PresetInfo(
            best_for  = "Troubleshooting issues",
            desc      = "Verbose logging & validation checks",
            emoji     = "🔍", 
            name      = "debug",
            overrides = PresetOverrides(overrides={
                "checkpoint.every_n_train_steps"    : 50,
                "experience.batch_size"             : 16,
                "experience.total_frames"           : 1_000,
                "flock.agent_count"                 : 3,
                "hardware.compile_model"            : False,
                "hardware.precision"                : "32-true",
                "metrics.enable_model_summary"      : True,
                "metrics.enable_progress_bar"       : True,
                "metrics.log_every_n_steps"         : 1,
                "metrics.profiler"                  : "simple",
                "optimizer.early_stopping_patience" : 3,
                "optimizer.gradient_clip_val"       : 10.0,
                "optimizer.learning_rate"           : 1e-3,
                "wandb.mode"                        : "offline"
            })
        ),
        description = "Debug preset with maximum visibility."
    )
    custom : PresetInfo = Field(
        default = PresetInfo(
            best_for  = "Advanced users",
            desc      = "Start from scratch with full control",
            emoji     = "🎯",
            name      = "custom",
            overrides = PresetOverrides()
        ),
        description = "Custom preset with no predefined overrides."
    )
