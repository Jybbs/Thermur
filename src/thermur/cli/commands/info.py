"""
System information command for the Thermur CLI.

Displays comprehensive system setup including hardware, software versions,
platform capabilities, and configuration details.
"""
from thermur.cli import app


def info():
    """
    📋 Display comprehensive system and configuration information.

    Shows detailed information about the current system setup, including
    hardware capabilities, software versions, and configuration status. This
    is useful for debugging, reporting issues, and understanding the
    environment in which the training is running.
    """
    app.ui.print_header("Thermur System Information")
    
    app.ui.print_section("System Information")
    info = app.system.get_system_info()
    app.ui.console.print(app.ui.create_system_table(info))
    app.ui.console.print()
    
    app.ui.display_wandb("info")
    app.ui.console.print()
    
    app.ui.print_section("Configuration System")
    app.ui.print_config_value(
        align_width = 11,
        key         = "Config Path",
        value       = "configs/"
    )
    
    app.ui.print_section("Common Commands")
    app.ui.print_command_examples(app.cfg.display.commands_examples)