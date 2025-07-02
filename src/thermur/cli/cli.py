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
    config  = cli_config,
    _parser = pydantic_parser
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
        self.system  = SystemInspector()
        self.ui      = ThermurUI(display_config = cfg.display)
        self.prompts = CLIPrompts(
            cli_cfg   = cfg.cli_config,
            messages  = cfg.messages,
            presets   = cfg.presets,
            prompts   = cfg.prompts,
            ui        = self.ui
        )


def create_cli():
    """
    Create and configure the Typer CLI application.
    """
    cli = Typer(
        context_settings = {"help_option_names": ["-h", "--help"]},
        help             = cfg.cli_config.app_description,
        name             = cfg.cli_config.app_name,
        rich_markup_mode = "rich",
    )
    
    cli.command(name="info")(info)
    cli.command(name="monitor")(monitor)
    cli.command(name="train")(train)
    cli.command(name="validate")(validate)
    
    cli.callback(invoke_without_command=True)(main_callback)
    
    return cli


def get_context(ctx: Context) -> AppContext:
    """
    Lazy context getter. Creates the AppContext on its first access.

    This ensures that the AppContext is only instantiated once per run,
    and its creation is deferred until it is actually needed by either the
    main callback or an eager callback like --version.

    Args:
        ctx : The Typer context.

    Returns:
        The singleton AppContext instance for the current application run.
    """
    if not ctx.obj:
        ctx.obj = AppContext()
        
    return ctx.obj


def main():
    """
    Main entry point for the CLI.
    
    This function creates and runs the CLI application with the loaded
    configuration. The context is handled by Typer during command invocation.
    """
    cli = create_cli()
    cli()


def main_callback(
    ctx     : Context,
    version : bool | None = Option(
        default     = None,
        param_decls = ["--version", "-v"],
        callback    = version_callback,
        is_eager    = True,
        help        = "Show version and system information",
    ),
):
    """
    🔥 Thermur: Thermally-constrained drone flock training toolkit.

    A command-line interface for training and managing thermal drone
    flock behaviors using imitation learning.

    Use 'thermur <command> --help' for detailed command information.
    """
    app_context = get_context(ctx = ctx)

    if ctx.invoked_subcommand is None:
        cfg = app_context.config
        ui  = app_context.ui

        ui.print_header(title = "Welcome to Thermur")
        ui.print_section(title = "Available Commands", style = "accent")

        for cmd_info in cfg.cli_config.commands_available:
            ui.console.print(
                f"  {cmd_info['icon']} [bold accent]{cmd_info['name']:10}"
                f"[/bold accent] [muted]{cmd_info['desc']}[/muted]"
            )

        ui.print_section(title = "Getting Started", style = "bright_green")

        for example in cfg.cli_config.commands_examples:
            ui.print_command_example(
                description = example["desc"],
                command     = example["command"],
                note        = example["note"]
            )

        ui.console.print()
        ui.print_message(message = cfg.messages.ready_to_train, style = "thermal")


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

    app_context = get_context(ctx = ctx)
    cfg         = app_context.config
    system      = app_context.system
    ui          = app_context.ui

    info = system.get_system_info(wandb_integration = cfg.wandb_integration)
    ui.console.print(f"thermur v{info['thermur']}")
    ui.console.print(f"Python v{info['python']} • PyTorch v{info['torch']}")

    raise Exit()


if __name__ == "__main__":
    main()