"""
Thermur command-line interface built with Typer and Rich.

This package provides the user-facing CLI for training and managing thermal drone
swarms. Commands are organized into subcommands for different workflows:
- train: Run imitation learning to train the GNN policy
- configure: Manage and explore configuration options
- info: Display system information and dependencies
- validate: Check configuration and environment setup
- monitor: Track training progress and system resources

The CLI uses lazy imports to maintain fast startup times for simple commands.
"""
from .cli      import AppContext, create_cli, main
from .commands import *
from .helpers  import *