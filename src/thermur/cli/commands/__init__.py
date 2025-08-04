"""
CLI command implementations.

Each command is implemented as a function that receives the AppContext through
Typer's dependency injection system and uses the shared UI components for output
formatting.
"""
from .download import *
from .info     import *
from .monitor  import *
from .runs     import *
from .train    import *
from .validate import *
