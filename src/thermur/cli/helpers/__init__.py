"""
Helper utilities for CLI functionality.

These classes provide the building blocks for the CLI's user experience:
- ThermurUI: Rich-based terminal UI with consistent theming and formatting
- SystemInspector: Hardware and dependency detection for compatibility checks
- CLIPrompts: Interactive prompts for configuration and user input
- ConfigExplorer: Navigation and exploration of the configuration tree

All helpers are initialized with the CLI configuration and share a common
visual style defined by the theme settings.
"""
from .explorer import ConfigExplorer
from .prompts  import CLIPrompts
from .system   import SystemInspector
from .ui       import ThermurUI