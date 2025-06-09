"""
Console-script target wired in pyproject.toml.

For now, it just prints runtime info so we can verify the install works 
end-to-end.
"""

from ..         import __version__
from __future__ import annotations
from rich       import print
from typing     import List

import sys

def cli(argv: List[str] | None = None):
    """
    Minimal CLI stub: replace with Typer commands later.
    """

    argv = argv or sys.argv[1:]
    if {"-v", "--version"} & set(argv):
        print(f"[bold green]Thermur[/] {__version__}")
        return
    
    print(f"[cyan]Thermur[/] {__version__} on Python {sys.version.split()[0]}")