"""
Enhanced console-script target for Thermur, built with Typer and Rich.

This module provides the main CLI interface by discovering and registering
all available commands from the .commands subpackage. It uses Hydra-zen
to load and validate the CLI configuration through Pydantic schemas.
"""
from .commands                      import *
from .helpers                       import *
from configs.cli                    import cli_config
from hydra_zen                      import instantiate
from hydra_zen.third_party.pydantic import pydantic_parser
from typer                          import Context, Exit, Option, Typer

cfg = instantiate(
    cli_config,
    _parser   = pydantic_parser,
    _convert_ = "all"
)


class AppContext:
    """
    A container for shared application state and components.

    This class initializes the core user interface, system inspector, and prompt
    orchestrator using the Hydra-zen configuration. An instance is created once
    in the main callback and passed to all commands via the Typer context.
    """
    def __init__(self):
        """
        Initialize the application context with the loaded configuration.
        """
        self.config  = cfg
        self.ui      = ThermurUI(cfg['theme'], cfg['ui'])
        self.system  = SystemInspector()
        self.prompts = CLIPrompts(self.ui, cfg['prompts'], cfg['messages'])


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
    cfg         = app_context.config
    system      = app_context.system

    ui.print_header(cfg['headers'].main_title, cfg['headers'].main_subtitle)

    info = system.get_system_info(cfg['wandb_display'])
    ui.print_config_value("Version", f"v{info['thermur']}", "Thermur package version")
    ui.print_config_value("Python",  f"v{info['python']}",  "Python runtime version")
    ui.print_config_value("PyTorch", f"v{info['torch']}",   "Deep learning framework")
    ui.console.print()

    table = ui.create_system_table(info)
    ui.console.print(table)

    ui.print_section(cfg['sections'].integration_status, style="swarm")
    status, details = system.check_wandb_status(cfg)
    ui.console.print(f"[swarm]📊 wandb: {status} • {details}[/swarm]")

    ui.print_section(cfg['sections'].quick_start, style="bright_green")

    for example in cfg['commands'].examples[:2]:
        ui.print_command_example(example["desc"], example["command"], example["note"])

    raise Exit()


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
        ui  = app_context.ui
        cfg = app_context.config

        ui.print_header(cfg['headers'].main_title, cfg['headers'].main_subtitle)
        ui.print_section(cfg['sections'].available_commands, "accent")

        for cmd_info in cfg['commands'].available:
            ui.console.print(
                f"  {cmd_info['icon']} [bold accent]{cmd_info['name']:10}"
                f"[/bold accent] [muted]{cmd_info['desc']}[/muted]"
            )

        ui.print_section(cfg['sections'].getting_started, "bright_green")

        for example in cfg['commands'].examples:
            ui.print_command_example(
                example["desc"],
                example["command"],
                example["note"]
            )

        ui.console.print()
        ui.print_message(cfg['messages'].ready_to_train, "thermal")


def create_cli():
    """
    Create and configure the Typer CLI application.
    """
    cli = Typer(
        name                     = cfg['cli'].app_name,
        help                     = cfg['cli'].app_description,
        add_completion           = False,
        rich_markup_mode         = "rich",
        no_args_is_help          = True,
        pretty_exceptions_enable = True,
    )
    
    cli.command(name="train")(train)
    cli.command(name="configure")(configure)
    cli.command(name="info")(info)
    cli.command(name="validate")(validate)
    cli.command(name="monitor")(monitor)
    
    cli.callback(invoke_without_command=True)(main_callback)
    
    return cli


def main():
    """
    Main entry point for the CLI.
    
    This function creates and runs the CLI application with the loaded
    configuration. The context is handled by Typer during command invocation.
    """
    cli = create_cli()
    cli()


if __name__ == "__main__":
    main()