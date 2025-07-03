"""
System information command for the Thermur CLI.

This module provides the 'info' command, which displays a comprehensive
overview of the system setup, including hardware, software versions,
platform capabilities, and configuration details.
"""
from typer import Context


def info(ctx: Context):
    """
    📋 Display comprehensive system and configuration information.

    Shows detailed information about the current system setup, including
    hardware capabilities, software versions, and configuration status. This
    is useful for debugging, reporting issues, and understanding the
    environment in which the training is running.
    """
    command = InfoCommand(ctx)
    command.run()


class InfoCommand:
    """
    Encapsulates the logic for the 'info' command.

    This class provides a structured way to gather and display system
    information, using components from the shared Typer context.
    """
    def __init__(self, ctx: Context):
        """
        Initializes the command with shared context components.

        Args:
            ctx: The Typer context, which holds the shared AppContext object
                 containing UI, system, and other core components.
        """
        self.cfg    = ctx.obj.cfg
        self.system = ctx.obj.system
        self.ui     = ctx.obj.ui

    def _perform_system_validation(self):
        """
        Performs comprehensive system validation checks.

        This helper validates hardware capabilities, software versions, and
        integration status, displaying the results in a formatted table.
        """
        self.ui.print_major_section("System Information")

        info = self.system.get_system_info()
        self.ui.console.print(self.ui.create_system_table(info))
        self.ui.console.print()

        status, details = self.system.check_wandb_status()
        self.ui.console.print(f"[flock]🎨 wandb: {status} • {details}[/flock]")
        self.ui.console.print()

    def run(self):
        """
        Executes the main workflow for displaying system info.

        This method orchestrates the process of printing headers, validating
        the system, and displaying tables of features and configurations.
        """
        self.ui.print_header("Thermur System Information")

        self._perform_system_validation()

        self.ui.print_major_section("Configuration System")
        self.ui.print_config_value(
            align_width = 11,
            key         = "Config Path",
            value       = "configs/"
        )
        
        preset_names = list(self.cfg.presets.presets.keys())
        self.ui.print_config_value(
            align_width = 11,
            key         = "Presets",
            value       = ", ".join(sorted(preset_names))
        )
        

        self.ui.print_major_section("Common Commands")
        for example in self.cfg.cli.commands_examples:
            self.ui.print_command_example(
                command     = example["command"],
                description = example["desc"],
                note        = example["note"]
            )