"""
Configuration explorer command for the Thermur CLI.

This module provides an interactive way for users to discover, navigate,
and edit configuration schemas, generating valid Hydra override strings
for use in other commands like 'train'.
"""
from ..helpers import ConfigExplorer
from typer     import Context, Typer

cmd_configure = Typer(
    add_completion           = False,
    rich_markup_mode         = "rich",
    no_args_is_help          = True,
    pretty_exceptions_enable = True,
)


@cmd_configure.command("configure")
def configure(ctx: Context):
    """
    🔧 Interactively explore and edit configurations.

    This command launches a terminal-based user interface that allows you to
    navigate the complete configuration hierarchy for the application. You can
    inspect Pydantic schemas, view default values, and edit parameters without
    needing to know the Hydra override syntax beforehand.

    The explorer will generate a list of override strings at the end, which can
    be copied and used directly with the `train` command.
    """
    command = ConfigureCommand(ctx)
    command.run()


class ConfigureCommand:
    """
    Encapsulates the logic for the 'configure' command.

    This class provides a structured way to manage the interactive
    configuration workflow, using components from the shared Typer context.
    """
    def __init__(self, ctx: Context):
        """
        Initializes the command with shared context components.

        Args:
            ctx: The Typer context, which holds the shared AppContext object
                 containing UI, prompts, and other core components.
        """
        self.constants = ctx.obj.constants
        self.prompts   = ctx.obj.prompts
        self.ui        = ctx.obj.ui
        self.explorer  = ConfigExplorer(self.ui, self.prompts, self.constants)

    def run(self):
        """
        Executes the main configuration exploration workflow.

        This method launches the interactive explorer and, if any changes are
        made, prints a summary of the generated command-line overrides for the
        user to copy.
        """
        overrides = self.explorer.explore_interactive()

        if overrides:
            self.ui.console.print()
            self.ui.print_header(self.constants.Headers.CONFIG_GEN_TITLE)

            self.ui.print_message(self.constants.Messages.CONFIG_GEN_ADD_CMD, "info")
            self.ui.console.print()

            cmd_parts = ["thermur train"]
            for override in overrides:
                cmd_parts.append(f'--config "{override}"')

            command = " ".join(cmd_parts)
            self.ui.console.print(f"[bold accent]$ {command}[/bold accent]")
            self.ui.console.print()

            self.ui.print_message(self.constants.Messages.CONFIG_GEN_USE_IND, "tip")
            for override in overrides:
                self.ui.print_config_value("--config", override)
        else:
            self.ui.print_message(self.constants.Messages.NO_CONFIG_CHANGES, "info")