"""
CLI command implementations.

Each command is implemented as a function that receives the AppContext through 
Typer's dependency injection system and uses the shared UI components for output 
formatting.
"""
from .data     import download, list, clean
from .info     import info
from .monitor  import monitor
from .train    import train
from .validate import validate