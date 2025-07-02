"""
Wandb monitoring command for the Thermur CLI.

This module provides the 'monitor' command, a convenient shortcut for
opening the Weights & Biases dashboard for a specified project in the
user's default web browser.
"""
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
        self.msgs   = self.config.messages
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
            project = self.config.wandb_display.default_project

        self.ui.print_header("wandb Monitoring")

        url = self.system.get_wandb_url(
            wandb_config = self.config.wandb_display, 
            ui_config    = self.config.ui, 
            project      = project
        )

        if not url:
            # Check if wandb is installed to provide a better error message
            info = self.system.get_system_info(self.config.wandb_display)
            if not info["wandb_installed"]:
                self.ui.print_message(
                    message  = "wandb is not installed in your Poetry environment",
                    msg_type = "error"
                )
                self.ui.print_message(
                    message  = "Run 'poetry install' to install all dependencies",
                    msg_type = "info"
                )
            else:
                # wandb is installed but user is not authenticated
                self.ui.print_message(
                    message  = self.msgs.wandb_unavailable,
                    msg_type = "error"
                )
                self.ui.print_message(
                    message  = "Run 'wandb login' to authenticate",
                    msg_type = "info"
                )
            raise Exit(1)

        self.ui.print_message(
            message  = self.msgs.browser_launch_template.format(project=project),
            msg_type = "flock"
        )
        self.ui.print_wandb_info(project, url)
        self.ui.console.print()

        try:
            with self.ui.console.status(
                self.config.status.launching_browser,
                spinner="dots"
            ):
                open(url)

            self.ui.print_message(
                message  = self.msgs.browser_success, 
                msg_type = "success"
            )
        except Exception as e:
            self.ui.print_message(
                message  = self.msgs.browser_fail_template.format(e=e), 
                msg_type = "error"
            )
            self.ui.print_message(
                message  = self.msgs.browser_manual_template.format(url=url),
                msg_type = "info"
            )