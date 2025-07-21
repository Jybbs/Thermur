"""
Wandb monitoring command for the Thermur CLI.

This module provides the 'monitor' command, a convenient shortcut for
opening the Weights & Biases dashboard for a specified project in the
user's default web browser.
"""
from typer      import Context, Exit
from webbrowser import open


def monitor(ctx: Context):
    """
    🎨 Open wandb monitoring dashboard for project configured in settings.

    Quickly opens the wandb dashboard in your default browser to monitor
    training progress, view metrics, and analyze experiment results.
    """
    command = MonitorCommand(ctx)
    command.run()


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
        self.cfg     = ctx.obj.cfg
        self.msgs    = self.cfg.messages
        self.prompts = ctx.obj.prompts
        self.system  = ctx.obj.system
        self.ui      = ctx.obj.ui

    def run(self):
        """
        Executes the main workflow for opening the wandb dashboard.

        This method uses the configured project name, generates the URL,
        and attempts to open it in the system's default web browser.
        """
        self.ui.print_header("wandb Monitoring")

        if url := self.ui.display_wandb("monitor", self.cfg.wandb.project):
            if self.prompts.confirm("Open browser to view dashboard?"):
                try:
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
        else:
            raise Exit(1)