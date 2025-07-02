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
    _parser   = pydantic_parser
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
        self.ui      = ThermurUI(cfg.theme, cfg.ui)
        self.system  = SystemInspector()
        self.prompts = CLIPrompts(self.ui, cfg.prompts, cfg.messages, cfg.commands)


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
    Shows version information.

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

    info = system.get_system_info(cfg.wandb_display)
    ui.console.print(f"thermur v{info['thermur']}")
    ui.console.print(f"Python v{info['python']} • PyTorch v{info['torch']}")

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
    🔥 Thermur: Thermally-constrained drone flock training toolkit.

    A command-line interface for training and managing thermal drone
    flock behaviors using imitation learning.

    Use 'thermur <command> --help' for detailed command information.
    """
    app_context = get_context(ctx)

    if ctx.invoked_subcommand is None:
        ui  = app_context.ui
        cfg = app_context.config

        ui.print_header("Welcome to Thermur")
        ui.print_section("Available Commands", "accent")

        for cmd_info in cfg.commands.available:
            ui.console.print(
                f"  {cmd_info['icon']} [bold accent]{cmd_info['name']:10}"
                f"[/bold accent] [muted]{cmd_info['desc']}[/muted]"
            )

        ui.print_section("Getting Started", "bright_green")

        for example in cfg.commands.examples:
            ui.print_command_example(
                example["desc"],
                example["command"],
                example["note"]
            )

        ui.console.print()
        ui.print_message(cfg.messages.ready_to_train, "thermal")


def create_cli():
    """
    Create and configure the Typer CLI application.
    """
    cli = Typer(
        name                     = cfg.cli.app_name,
        help                     = cfg.cli.app_description,
        rich_markup_mode         = "rich",
        context_settings         = {"help_option_names": ["-h", "--help"]},
    )
    
    cli.command(name="train")(train)
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