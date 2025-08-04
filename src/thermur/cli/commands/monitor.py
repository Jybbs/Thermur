"""
Wandb monitoring command for the Thermur CLI.

Provides a convenient shortcut for opening the Weights & Biases dashboard
for a specified project in the user's default web browser.
"""
from thermur.cli import app
from typer       import Exit
from webbrowser  import open


def monitor():
    """
    🎨 Open wandb monitoring dashboard for project configured in settings.

    Quickly opens the wandb dashboard in your default browser to monitor
    training progress, view metrics, and analyze experiment results.
    """
    app.ui.print_header("wandb Monitoring")

    if url := app.ui.display_wandb("monitor", app.cfg.wandb.project):
        if app.prompts.confirm("Open browser to view dashboard?"):
            try:
                open(url)
                app.ui.print_message(
                    message  = "Dashboard opened in your default browser!",
                    msg_type = "success"
                )
            except Exception as e:
                app.ui.print_message(
                    message  = f"Failed to open browser: {e}",
                    msg_type = "error"
                )
                app.ui.print_message(
                    message  = f"Please visit manually: {url}",
                    msg_type = "info"
                )
    else:
        raise Exit(1)
