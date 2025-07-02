"""
Helper utilities for CLI functionality.

These classes provide the building blocks for the CLI's user experience:
- ThermurUI       : Rich-based terminal UI with consistent theming and formatting
- SystemInspector : Hardware and dependency detection for compatibility checks
- CLIPrompts      : Interactive prompts for configuration and user input

All helpers are initialized with the CLI configuration and share a common
visual style defined by the theme settings.
"""
from .prompts  import CLIPrompts
from .system   import SystemInspector
from .ui       import ThermurUI