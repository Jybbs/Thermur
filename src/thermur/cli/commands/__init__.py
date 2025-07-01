"""
CLI command implementations as Typer sub-applications.

Each command is implemented as a separate Typer instance that gets mounted to
the main CLI. This modular structure allows commands to have their own options,
callbacks, and help text while maintaining a cohesive user experience.

Commands follow a consistent pattern: they receive the AppContext through Typer's
dependency injection system and use the shared UI components for output formatting.
"""
from .configure import cmd_configure
from .info      import cmd_info
from .monitor   import cmd_monitor
from .train     import cmd_train
from .validate  import cmd_validate