"""
Entry-point shim so that:

    python -m thermur --version

works even if the console script isn't installed (e.g. in editable mode).
"""

from core.cli import cli

if __name__ == "__main__":

    cli()