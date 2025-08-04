"""
Enhanced console-script target for Thermur, built with Typer and Rich.

This module provides the main CLI interface by discovering and registering
all available commands from the .commands subpackage.
"""
from operator             import itemgetter
from thermur.cli          import app
from thermur.cli.commands import *
from typer                import Context, Exit, Option, Typer

cli = Typer(
    context_settings = {"help_option_names": ["-h", "--help"]},
    help             = (
        "🔥 Thermur: Advanced thermal-aware drone flock training "
        "using imitation learning, physics-based constraints, and "
        "real-time monitoring"
    ),
    name             = "thermur",
    rich_markup_mode = "rich"
)

cli.command()(download)
cli.command()(info)
cli.command()(monitor)
cli.command()(train)
cli.command()(validate)
cli.add_typer(runs, name="runs")


@cli.callback(invoke_without_command=True)
def main_callback(
    ctx     : Context,
    version : bool | None = Option(
        None,
        "--version", "-v",
        callback = None,
        help     = "Show version and system information",
        is_eager = True
    )
):
    """
    🔥 Thermur: Thermally-constrained drone flock training toolkit.

    A command-line interface for training and managing thermal drone
    flock behaviors using imitation learning.

    Use 'thermur <command> --help' for detailed command information.
    """
    if version:
        info = app.system.get_system_info()
        thermur, python, torch = itemgetter('thermur', 'python', 'torch')(info)
        app.ui.console.print(f"thermur v{thermur}")
        app.ui.console.print(f"Python v{python} • PyTorch v{torch}")
        raise Exit()

    if ctx.invoked_subcommand is None:
        app.ui.print_header("Welcome to Thermur")
        app.ui.print_section("Available Commands", style="accent")

        for cmd_info in app.cfg.display.commands_available:
            app.ui.console.print(
                f"  {cmd_info['icon']} [bold accent]{cmd_info['name']:10}"
                f"[/bold accent] [muted]{cmd_info['desc']}[/muted]"
            )

        app.ui.print_section("Getting Started", style="bright_green")
        app.ui.print_command_examples(app.cfg.display.commands_examples)

        app.ui.console.print()
        app.ui.print_message(
            message  = "Ready to train some thermal flocks? 🔥",
            msg_type = "thermal"
        )


def main():
    """
    Entry point for the thermur console script.
    """
    cli()
