"""
System information command for the Thermur CLI.

This module provides the 'info' command, which displays a comprehensive
overview of the system setup, including hardware, software versions,
platform capabilities, and configuration details.
"""
from typer import Context, Typer

cmd_info = Typer(
    add_completion           = False,
    rich_markup_mode         = "rich",
    no_args_is_help          = True,
    pretty_exceptions_enable = True,
)


@cmd_info.command("info")
def info(ctx: Context):
    """
    📊 Display comprehensive system and configuration information.

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
        self.constants = ctx.obj.constants
        self.system    = ctx.obj.system
        self.ui        = ctx.obj.ui

    def run(self):
        """
        Executes the main workflow for displaying system info.

        This method orchestrates the process of printing headers, validating
        the system, and displaying tables of features and configurations.
        """
        self.ui.print_header(self.constants.Headers.INFO_TITLE)

        self._perform_system_validation()

        self.ui.print_section(self.constants.Sections.FEATURES, "accent")
        features_table = self.ui.create_feature_table()
        self.ui.console.print(features_table)

        self.ui.print_section(self.constants.Sections.CONFIG_SYSTEM, "config")
        self.ui.print_config_value(
            "Config Path",
            "configs/",
            "Hydra configuration directory"
        )

        preset_names = ", ".join(self.constants.Presets.CONFIGS.keys())
        self.ui.print_config_value(
            "Presets",
            preset_names,
            "Available training presets"
        )

        self.ui.print_config_value(
            "Explorer",
            "thermur configure",
            "Interactive configuration tool"
        )

        self.ui.print_training_tips()

    def _perform_system_validation(self):
        """
        Performs comprehensive system validation checks.

        This helper validates hardware capabilities, software versions, and
        integration status, displaying the results in a formatted table.
        """
        self.ui.print_section(self.constants.Sections.SYSTEM_VALIDATION, "thermal")

        info = self.system.get_system_info(self.constants)
        table = self.ui.create_system_table(info)

        self.ui.console.print(table)
        self.ui.console.print()

        status, details = self.system.check_wandb_status(self.constants)
        self.ui.console.print(f"[swarm]📊 wandb: {status} • {details}[/swarm]")
        self.ui.console.print()