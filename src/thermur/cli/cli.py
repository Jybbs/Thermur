"""
Enhanced console-script target for Thermur, built with Typer and Rich.

This module provides the main CLI interface by discovering and registering
all available commands from the .commands subpackage. It is also responsible
for creating the shared application context.
"""
from .commands  import *
from .constants import CLIConstants
from .prompts   import CLIPrompts
from .system    import SystemInspector
from .ui        import ThermurUI
from typer      import Context, Exit, Option, Typer


class AppContext:
    """
    A container for shared application state and components.

    This class initializes the core user interface, constants, system inspector,
    and prompt orchestrator a single time. An instance of this class is created
    in the main callback and passed to all commands via the Typer context.
    """
    def __init__(self):
        self.constants = CLIConstants()
        self.ui        = ThermurUI(self.constants)
        self.system    = SystemInspector()
        self.prompts   = CLIPrompts(self.ui, self.constants)


def get_context(ctx: Context) -> AppContext:
    """
    Lazy context getter. Creates the AppContext on its first access.

    This ensures that the AppContext is only instantiated once per run,
    and its creation is deferred until it is actually needed by either the
    main callback or an eager callback like --version.

    Args:
        ctx: The Typer context.

    Returns:
        The singleton AppContext instance for the current application run.
    """
    if not ctx.obj:
        ctx.obj = AppContext()
        
    return ctx.obj


cli = Typer(
    name                     = "thermur",
    help                     = "🔥 Thermally-constrained drone swarm training toolkit",
    add_completion           = False,
    rich_markup_mode         = "rich",
    no_args_is_help          = True,
    pretty_exceptions_enable = True,
)

cli.add_typer(cmd_train,     name = "train")
cli.add_typer(cmd_configure, name = "configure")
cli.add_typer(cmd_info,      name = "info")
cli.add_typer(cmd_validate,  name = "validate")
cli.add_typer(cmd_monitor,   name = "monitor")


def version_callback(value: bool, ctx: Context):
    """
    Shows version information and system details.

    This is an "eager" callback that runs before the main callback. It uses
    the lazy context getter to ensure the AppContext is only created once.

    Args:
        value : True if the --version flag is present.
        ctx   : The Typer context.
    """
    if not value:
        return

    app_context = get_context(ctx)
    ui          = app_context.ui
    constants   = app_context.constants
    system      = app_context.system

    ui.print_header(constants.Headers.MAIN_TITLE, constants.Headers.MAIN_SUBTITLE)

    info = system.get_system_info(constants)
    ui.print_config_value("Version", f"v{info['thermur']}", "Thermur package version")
    ui.print_config_value("Python",  f"v{info['python']}",  "Python runtime version")
    ui.print_config_value("PyTorch", f"v{info['torch']}",   "Deep learning framework")
    ui.console.print()

    table = ui.create_system_table(info)
    ui.console.print(table)

    ui.print_section(constants.Sections.INTEGRATION_STATUS, style="swarm")
    status, details = system.check_wandb_status(constants)
    ui.console.print(f"[swarm]📊 wandb: {status} • {details}[/swarm]")

    ui.print_section(constants.Sections.QUICK_START, style="bright_green")

    for example in constants.Commands.EXAMPLES[:2]:
        ui.print_command_example(example["desc"], example["command"], example["note"])

    raise Exit()


@cli.callback(invoke_without_command=True)
def main_callback(
    ctx     : Context,
    version : bool | None = Option(
        None,
        "--version",
        "-v",
        callback = version_callback,
        is_eager = True,
        help     = "Show version and system information",
    ),
):
    """
    🔥 Thermur: Thermally-constrained drone swarm training toolkit.

    A command-line interface for training and managing thermal drone
    swarm behaviors using imitation learning.

    Use 'thermur <command> --help' for detailed command information.
    """
    app_context = get_context(ctx)

    if ctx.invoked_subcommand is None:
        ui = app_context.ui
        constants = app_context.constants

        ui.print_header(constants.Headers.MAIN_TITLE, constants.Headers.MAIN_SUBTITLE)
        ui.print_section(constants.Sections.AVAILABLE_COMMANDS, "accent")

        for cmd_info in constants.Commands.AVAILABLE:
            ui.console.print(
                f"  {cmd_info['icon']} [bold accent]{cmd_info['name']:10}"
                f"[/bold accent] [muted]{cmd_info['desc']}[/muted]"
            )

        ui.print_section(constants.Sections.GETTING_STARTED, "bright_green")

        for example in constants.Commands.EXAMPLES:
            ui.print_command_example(
                example["desc"],
                example["command"],
                example["note"]
            )

        ui.console.print()
        ui.print_message(constants.Messages.READY_TO_TRAIN, "thermal")