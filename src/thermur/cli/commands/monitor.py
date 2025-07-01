"""
Wandb monitoring command for the Thermur CLI.

This module provides the 'monitor' command, a convenient shortcut for
opening the Weights & Biases dashboard for a specified project in the
user's default web browser.
"""
from time       import sleep
from typer      import Context, Exit, Option
from webbrowser import open


def monitor(
    ctx     : Context,
    project : str | None = Option(
        None,
        "--project", "-p",
        help="wandb project name to monitor. Defaults to 'thermur'."
    ),
):
    """
    🎨 Open wandb monitoring dashboard for the specified project.

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
        self.config = ctx.obj.config
        self.system = ctx.obj.system
        self.ui     = ctx.obj.ui

    def run(self, project: str | None):
        """
        Executes the main workflow for opening the wandb dashboard.

        This method resolves the project name if not provided, generates the
        URL, and attempts to open it in the system's default web browser.

        Args:
            project : The wandb project name to open, or None to use the default.
        """
        if project is None:
            project = self.config.wandb.default_project

        self.ui.print_header(self.config.headers.monitor_title)

        url = self.system.get_wandb_url(self.config, project)

        if not url:
            self.ui.print_message(
                self.config.messages.wandb_unavailable,
                "error"
            )
            raise Exit(1)

        self.ui.print_message(
            self.config.messages.browser_launch_tpl.format(project=project),
            "flock"
        )
        self.ui.print_wandb_info(project, url)
        self.ui.console.print()

        try:
            with self.ui.console.status(
                self.config.status.launching_browser,
                spinner="dots"
            ):
                sleep(0.5)
                open(url)
            self.ui.print_message(self.config.messages.browser_success, "success")
        except Exception as e:
            self.ui.print_message(self.config.messages.browser_fail_tpl.format(e=e), "error")
            self.ui.print_message(self.config.messages.browser_manual_tpl.format(url=url), "info")