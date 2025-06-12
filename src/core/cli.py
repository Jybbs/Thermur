"""
Console-script target wired in pyproject.toml, built with Typer.

The main entry point is the `app` object. Commands are registered to it using
the `@app.command()` decorator.

Execution Flow:
1. User runs `thermur` or `python -m src` in the terminal.
2. The `[project.scripts]` entry in `pyproject.toml` points to `src.__main__:cli`.
3. `src.__main__.py` calls the `app` object from this file.
4. `typer` parses the command-line arguments (e.g., `train`, `--version`).
5. `typer` invokes the appropriate function (e.g., `train()` or `version_callback`).
"""
from ..         import __version__
from __future__ import annotations
from rich       import print
from typer      import Context, Exit, Option, Typer
from typing     import Optional


# The docstring of this object will be used as the main `--help` text
app = Typer(
    name           = "thermur",
    help           = "A toolkit for simulating thermally-constrained drone swarms.",
    add_completion = False
)

def version_callback(value: bool):
    """
    A callback function triggered by the `--version` flag.

    `typer` invokes this function if '--version' or '-v' is present on the
    command line. If it is, we print the version and then call `typer.Exit()`
    to cleanly terminate the program before any other logic runs.

    Args:
        value: The boolean value of the flag (True if present).
    """
    if value:
        print(f"[bold green]Thermur[/] version: {__version__}")
        raise Exit()

@app.command()
def train():
    """
    Launches a new training run.

    This command serves as the entry point to the training script. It performs a
    lazy, local import of the `train_main` function. 
    
    This keeps the CLI itself lightweight and fast, as heavyweight libraries like 
    `torch` and `hydra` are only imported when this specific command is actually 
    executed, not when just asking for `--help` or `--version`.
    """
    print("[yellow]CLI forwarding to training script...[/]")
    print("[grey70]Hydra will now take over to manage configuration.[/]")

    raise NotImplementedError("Data interpolation logic to be implemented.")

@app.callback()
def main_callback(
    ctx     : Context,
    version : Optional[bool] = Option(
        None,
        "--version",
        "-v",
        callback = version_callback,
        is_eager = True,
        help     = "Show the application's version and exit.",
    ),
):
    """
    The main application callback, run before any command.

    This function is automatically invoked by Typer before any specific command
    (like `train`) is run. It's the ideal place for global flags like
    `--version` that should apply to the application as a whole.
    """
    pass