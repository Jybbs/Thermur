"""
Wandb monitoring command for the Thermur CLI.

This module provides the 'monitor' command, a convenient shortcut for
opening the Weights & Biases dashboard for a specified project in the
user's default web browser.
"""
from time       import sleep
from typer      import Context, Exit, Option, Typer
from webbrowser import open

cmd_monitor = Typer(
    add_completion           = False,
    rich_markup_mode         = "rich",
    no_args_is_help          = True,
    pretty_exceptions_enable = True,
)


@cmd_monitor.command("monitor")
def monitor(
    ctx     : Context,
    project : str | None = Option(
        None,
        "--project", "-p",
        help="wandb project name to monitor. Defaults to 'thermur'."
    ),
):
    """
    📈 Open wandb monitoring dashboard for the specified project.

    Quickly opens the wandb dashboard in your default browser to monitor
    training progress, view metrics, and analyze experiment results.
    """
    command = MonitorCommand(ctx)
    command.run(project=project)


class MonitorCommand:
    """
    Encapsulates the logic for the 'monitor' command.

    This class provides a structured way to get the wandb URL and open it,
    using components from the shared Typer context.
    """
    def __init__(self, ctx: Context):
        """
        Initializes the command with shared context components.

        Args:
            ctx: The Typer context, which holds the shared AppContext object
                 containing UI, system, and other core components.
        """
        self.constants = ctx.obj.constants
        self.system    = ctx.obj.system
        self.ui        = ctx.obj.ui

    def run(self, project: str | None):
        """
        Executes the main workflow for opening the wandb dashboard.

        This method resolves the project name if not provided, generates the
        URL, and attempts to open it in the system's default web browser.

        Args:
            project : The wandb project name to open, or None to use the default.
        """
        if project is None:
            project = self.constants.Wandb.DEFAULT_PROJECT

        self.ui.print_header(
            self.constants.Headers.MONITOR_TITLE,
            self.constants.Headers.MONITOR_SUBTITLE_TPL.format(project=project)
        )

        url = self.system.get_wandb_url(self.constants, project)

        if not url:
            self.ui.print_message(
                self.constants.Messages.WANDB_UNAVAILABLE,
                "error"
            )
            raise Exit(1)

        self.ui.print_message(
            self.constants.Messages.BROWSER_LAUNCH_TPL.format(project=project),
            "swarm"
        )
        self.ui.print_wandb_info(project, url)
        self.ui.console.print()

        try:
            with self.ui.console.status(
                self.constants.Status.LAUNCHING_BROWSER,
                spinner="dots"
            ):
                sleep(0.5)
                open(url)
            self.ui.print_message(self.constants.Messages.BROWSER_SUCCESS, "success")
        except Exception as e:
            self.ui.print_message(self.constants.Messages.BROWSER_FAIL_TPL.format(e=e), "error")
            self.ui.print_message(self.constants.Messages.BROWSER_MANUAL_TPL.format(url=url), "info")