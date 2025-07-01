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
        self.config = ctx.obj.config
        self.system = ctx.obj.system
        self.ui     = ctx.obj.ui

    def run(self):
        """
        Executes the main workflow for displaying system info.

        This method orchestrates the process of printing headers, validating
        the system, and displaying tables of features and configurations.
        """
        self.ui.print_header(self.config.headers.info_title)

        self._perform_system_validation()

        self.ui.print_section(self.config.sections.features, "accent")
        features_table = self.ui.create_feature_table()
        self.ui.console.print(features_table)

        self.ui.print_section(self.config.sections.config_system, "config")
        self.ui.print_config_value("Config Path", "configs/", align_width=11)
        
        preset_fields = set(self.config.presets.__fields__) - {"table_title"}
        self.ui.print_config_value("Presets", ", ".join(sorted(preset_fields)), align_width=11)
        
        self.ui.print_config_value("Explorer", "thermur configure", align_width=11)

        self.ui.print_section(self.config.sections.common_commands, style="bright_green")
        for example in self.config.commands.examples:
            self.ui.print_command_example(example["desc"], example["command"], example["note"])

    def _perform_system_validation(self):
        """
        Performs comprehensive system validation checks.

        This helper validates hardware capabilities, software versions, and
        integration status, displaying the results in a formatted table.
        """
        self.ui.print_section(self.config.sections.system_validation, "thermal")

        info  = self.system.get_system_info(self.config.wandb_display)
        table = self.ui.create_system_table(info, self.config.system)

        self.ui.console.print(table)
        self.ui.console.print()

        status, details = self.system.check_wandb_status(self.config)
        self.ui.console.print(f"[flock]🪄  wandb: {status} • {details}[/flock]")
        self.ui.console.print()