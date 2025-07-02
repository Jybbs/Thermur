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
    _parser = pydantic_parser,
    config  = cli_config
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
        self.ui      = ThermurUI(cfg.display)
        self.prompts = CLIPrompts(
            cli_cfg  = cfg.cli_config,
            messages = cfg.messages,
            presets  = cfg.presets,
            prompts  = cfg.prompts,
            ui       = self.ui
        )


class ThermurCLI:
    """
    Main CLI application class for Thermur.
    
    This class encapsulates all CLI-related functionality, including command
    registration, callbacks, and the main entry point.
    """
    
    def __init__(self):
        """
        Initialize the CLI with configuration.
        """
        self.cfg = cfg
        self.cli = self._create_cli()
    
    def _create_cli(self) -> Typer:
        """
        Create and configure the Typer CLI application.
        """
        cli = Typer(
            context_settings = {"help_option_names": ["-h", "--help"]},
            help             = self.cfg.cli_config.app_description,
            name             = self.cfg.cli_config.app_name,
            rich_markup_mode = "rich"
        )
        
        cli.command(name="info")(info)
        cli.command(name="monitor")(monitor)
        cli.command(name="train")(train)
        cli.command(name="validate")(validate)
        
        cli.callback(invoke_without_command=True)(self._main_callback)
        
        return cli
    
    def _get_context(self, ctx: Context) -> AppContext:
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
    
    def _main_callback(
        self,
        ctx     : Context,
        version : bool | None = Option(
            callback    = None,
            default     = None,
            help        = "Show version and system information",
            is_eager    = True,
            param_decls = ["--version", "-v"]
        )
    ):
        """
        🔥 Thermur: Thermally-constrained drone flock training toolkit.

        A command-line interface for training and managing thermal drone
        flock behaviors using imitation learning.

        Use 'thermur <command> --help' for detailed command information.
        """
        if version:
            self._version_callback(version, ctx)
        
        app_context = self._get_context(ctx = ctx)

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
                    command     = example["command"],
                    description = example["desc"],
                    note        = example["note"]
                )

            ui.console.print()
            ui.print_message(
                message  = cfg.messages.ready_to_train, 
                msg_type = "thermal"
            )
    
    def _version_callback(self, value: bool, ctx: Context):
        """
        Shows version information.

        This is an "eager" callback that runs before the main callback. It uses
        the lazy context getter to ensure the AppContext is only created once.

        Args:
            ctx   : The Typer context.
            value : True if the --version flag is present.
        """
        if not value:
            return

        app_context = self._get_context(ctx = ctx)
        cfg         = app_context.config
        system      = app_context.system
        ui          = app_context.ui

        info = system.get_system_info(cfg.wandb_integration)
        ui.console.print(f"thermur v{info['thermur']}")
        ui.console.print(f"Python v{info['python']} • PyTorch v{info['torch']}")

        raise Exit()
    
    def run(self):
        """
        Run the CLI application.
        """
        self.cli()


def main():
    """
    Main entry point for the CLI.
    """
    app = ThermurCLI()
    app.run()


if __name__ == "__main__":
    main()