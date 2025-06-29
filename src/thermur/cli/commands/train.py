"""
Training command for the Thermur CLI.

This module encapsulates all logic for the 'train' command, including
system validation, configuration, and the initialization of the
imitation learning workflow.
"""
from sys   import argv
from time  import sleep
from typer import Context, Exit, Option, Typer

cmd_train = Typer(
    add_completion           = False,
    rich_markup_mode         = "rich",
    no_args_is_help          = True,
    pretty_exceptions_enable = True,
)


@cmd_train.command("train")
def train(
    ctx: Context,
    config_overrides: list[str] | None = Option(
        None,
        "--config", "-c",
        help="Hydra configuration overrides"
    ),
    force: bool = Option(
        False,
        "--force", "-f",
        help="Skip system checks and warnings"
    ),
    interactive: bool = Option(
        True,
        "--interactive/--no-interactive", "-i/-n",
        help="Enable interactive configuration prompts"
    ),
    preset: str | None = Option(
        None,
        "--preset", "-p",
        help="Configuration preset (quick, standard, large, debug)"
    ),
    wandb_project: str | None = Option(
        None,
        "--wandb-project", "-w",
        help="wandb project name for experiment tracking"
    ),
):
    """
    🔥 Train the thermal drone swarm using imitation learning.

    This command provides a comprehensive training workflow with system
    validation, configuration management, and seamless wandb integration.

    Examples:
        thermur train                                   # Interactive training
        thermur train --preset quick                    # Quick test run
        thermur train --config hyperparameters.lr=0.01  # Custom learning rate
        thermur train --wandb-project my-experiment     # Custom wandb project
        thermur train --no-interactive --force          # Non-interactive mode
    """
    command = TrainCommand(ctx)
    command.run(
        config_overrides = config_overrides,
        force            = force,
        interactive      = interactive,
        preset           = preset,
        wandb_project    = wandb_project,
    )


class TrainCommand:
    """
    Encapsulates the state and logic for the 'train' command.

    This class provides a structured way to manage the training workflow,
    holding shared components from the Typer context and organizing the
    process into a series of well-defined, testable methods.
    """
    def __init__(self, ctx: Context):
        """
        Initializes the command with shared context components.

        Args:
            ctx: The Typer context, which holds the shared AppContext object
                 containing UI, system, and other core components.
        """
        self.constants = ctx.obj.constants
        self.prompts   = ctx.obj.prompts
        self.system    = ctx.obj.system
        self.ui        = ctx.obj.ui

    def run(
        self,
        config_overrides : list[str] | None,
        force            : bool,
        interactive      : bool,
        preset           : str | None,
        wandb_project    : str | None,
    ):
        """
        Executes the main training workflow from start to finish.

        This method orchestrates the entire training process, including
        validation, configuration, user confirmation, and the final
        launch of the core training logic.

        Args:
            config_overrides : A list of Hydra configuration overrides.
            force            : If True, skips system validation checks.
            interactive      : If True, enables interactive prompts for configuration.
            preset           : The name of the configuration preset to use.
            wandb_project    : The name of the Weights & Biases project for tracking.
        """
        self.ui.print_header(
            self.constants.Headers.TRAIN_TITLE,
            self.constants.Headers.TRAIN_SUBTITLE
        )

        if not force:
            self._perform_system_validation()
        else:
            self.ui.print_message(self.constants.Messages.SKIPPING_CHECKS, "warning")

        preset, wandb_project, config_overrides = self._setup_configuration(
            config_overrides = config_overrides,
            force            = force,
            interactive      = interactive,
            preset           = preset,
            wandb_project    = wandb_project,
        )

        if interactive:
            self._confirm_training_setup(
                config_overrides = config_overrides,
                preset           = preset,
                wandb_project    = wandb_project
            )

        self._initiate_training(
            config_overrides = config_overrides,
            preset           = preset,
            wandb_project    = wandb_project
        )

    def _build_hydra_configuration(
        self,
        config_overrides : list[str] | None,
        preset           : str | None,
        wandb_project    : str,
    ):
        """
        Builds the Hydra-decorated training function with lazy imports.

        This method wraps the core training logic in a function that is then
        decorated by Hydra. Heavy dependencies like PyTorch and Hydra itself
        are imported only when this method is called, ensuring that other CLI
        commands (like `info` or `validate`) have a fast startup time.

        Args:
            config_overrides : A list of configuration override strings.
            preset           : The configuration preset name.
            wandb_project    : The wandb project name for tracking.

        Returns:
            A Hydra-decorated function ready to be executed.
        """
        with self.ui.create_thermal_progress() as progress:
            task = progress.add_task(self.constants.Status.INIT_MODULES, total=100)

            progress.update(
                task,
                advance     = 20,
                description = self.constants.Status.LOADING_CONFIG_SYS
            )
            from configs                        import imitation_config, register_configs
            from hydra_zen                      import instantiate, zen
            from hydra_zen.third_party.pydantic import pydantic_parser
            from thermur                        import (
                configure_loguru,
                set_seed,
                train_imitation_learning
            )

            progress.update(
                task,
                advance     = 30,
                description = self.constants.Status.REGISTERING_CONFIGS
            )
            register_configs()

            progress.update(
                task,
                advance     = 50,
                description = self.constants.Status.PREPARING_HYDRA
            )

            @zen(imitation_config).hydra_main(
                config_name            = "train",
                config_path            = None,
                version_base           = None,
                with_log_configuration = False,
            )
            def hydra_train(cfg):
                """
                Executes training with a fully resolved Hydra configuration.

                This inner function is the target of the Hydra decorator. It
                receives the populated configuration object (`cfg`) and orchestrates
                the final training workflow.

                Args:
                    cfg : The fully instantiated and merged Hydra config object.
                """
                self._execute_training_workflow(
                    cfg                      = cfg,
                    configure_loguru         = configure_loguru,
                    instantiate              = instantiate,
                    pydantic_parser          = pydantic_parser,
                    set_seed                 = set_seed,
                    train_imitation_learning = train_imitation_learning,
                )

            progress.update(task, advance=100, description=self.constants.Status.READY_TO_TRAIN)

        return hydra_train

    def _confirm_training_setup(
        self,
        config_overrides : list[str] | None,
        preset           : str | None,
        wandb_project    : str,
    ):
        """
        Shows a final training summary and asks the user for confirmation.

        This method presents a consolidated view of all major training settings
        and acts as the final gate before initiating the potentially long-running
        training process, giving the user a chance to cancel.

        Args:
            config_overrides : The list of custom configuration overrides.
            preset           : The selected configuration preset.
            wandb_project    : The configured wandb project name.
        """
        info = self.system.get_system_info(self.constants)
        summary = {
            "preset"        : preset or "default",
            "wandb_project" : wandb_project,
            "overrides"     : len(config_overrides or []),
            "gpu_available" : info["cuda"],
        }

        if not self.prompts.show_training_summary(summary):
            self.ui.print_message(self.constants.Messages.TRAINING_CANCELLED, "warning")
            raise Exit()

    def _execute_training_workflow(
        self,
        cfg,
        configure_loguru,
        instantiate,
        pydantic_parser,
        set_seed,
        train_imitation_learning,
    ):
        """
        Executes the main training workflow with instantiated components.

        This function takes the resolved Hydra configuration and uses it to set up
        logging, random seeds, and instantiate all necessary training components
        before starting the main training loop.

        Args:
            cfg                      : The resolved Hydra configuration object.
            configure_loguru         : The function to configure the logger.
            instantiate              : The hydra-zen function to instantiate objects.
            pydantic_parser          : The parser for Pydantic models.
            set_seed                 : The function to set the random seed.
            train_imitation_learning : The core function for the training loop.
        """
        self.ui.print_section(self.constants.Sections.BUILDING_COMPONENTS, "thermal")

        configure_loguru(
            instantiate(cfg.monitoring.logging, _parser=pydantic_parser)
        )
        set_seed(
            instantiate(cfg.hyperparameters, _parser=pydantic_parser).seed
        )

        components = self._instantiate_training_components(
            cfg             = cfg,
            instantiate     = instantiate,
            pydantic_parser = pydantic_parser,
        )

        self.ui.print_message(self.constants.Messages.COMPONENTS_INITIALIZED, "success")
        self.ui.console.print()

        self.ui.print_section(self.constants.Sections.TRAINING_STARTED, "thermal")
        self.ui.print_message(self.constants.Messages.MONITORING_DYNAMICS, "thermal")
        self.ui.print_message(self.constants.Messages.TRACK_WANDB, "swarm")
        self.ui.console.print()

        train_imitation_learning(**components)

        self.ui.console.print()
        self.ui.print_header(
            self.constants.Messages.TRAINING_COMPLETE_HEADER,
            self.constants.Messages.TRAINING_COMPLETE_SUB
        )

    def _handle_configuration_issues(
        self,
        interactive : bool,
        issues      : list[str],
    ):
        """
        Handles reported configuration validation issues.

        If in interactive mode, it displays the issues and asks the user for
        confirmation to proceed. In non-interactive mode, it prints the errors
        and exits the application with a non-zero status code.

        Args:
            interactive : Whether the CLI is in interactive mode.
            issues      : A list of validation issue strings to display.
        """
        if interactive and not self.prompts.confirm_system_override(issues):
            self.ui.print_message(self.constants.Messages.TRAINING_CANCELLED, "warning")
            raise Exit()

        elif not interactive:
            self.ui.print_message(self.constants.Validation.CONFIG_FAIL_MSG, "error")

            for issue in issues:
                self.ui.console.print(f"  • {issue}")
            self.ui.print_message(
                self.constants.Validation.FORCE_OVERRIDE_TIP,
                "info"
            )
            raise Exit(1)

    def _initiate_training(
        self,
        config_overrides : list[str] | None,
        preset           : str | None,
        wandb_project    : str,
    ):
        """
        Initializes and starts the training process, handling exceptions.

        This method serves as the final gateway before running the core training
        logic. It prints final status messages and wraps the call in a
        try/except block to gracefully handle user interruptions and errors.

        Args:
            config_overrides : The list of custom configuration overrides.
            preset           : The selected configuration preset.
            wandb_project    : The configured wandb project name.
        """
        self.ui.print_section(self.constants.Sections.INIT_TRAINING, "thermal")

        url = self.system.get_wandb_url(self.constants, wandb_project)
        self.ui.print_wandb_info(wandb_project, url)
        self.ui.console.print()

        try:
            self._run_training(config_overrides, preset, wandb_project)

        except KeyboardInterrupt:
            self.ui.console.print()
            self.ui.print_message(self.constants.Messages.TRAINING_INTERRUPTED, "warning")
            raise Exit()

        except Exception as e:
            self.ui.print_message(self.constants.Messages.TRAINING_FAILED_TPL.format(e=e), "error")
            raise Exit(1)

    def _instantiate_training_components(
        self,
        cfg,
        instantiate,
        pydantic_parser
    ):
        """
        Instantiates all required training components from the configuration.

        This method iterates through a predefined list of components, resolves
        their configurations from the main `cfg` object, and instantiates
        them using hydra-zen. A progress bar provides visual feedback.

        Args:
            cfg             : The resolved Hydra configuration object.
            instantiate     : The hydra-zen function to instantiate objects.
            pydantic_parser : The parser for Pydantic models.

        Returns:
            A dictionary of the instantiated training components.
        """
        with self.ui.create_thermal_progress() as progress:
            component_configs = self.constants.Training.COMPONENT_CONFIGS
            task = progress.add_task(
                self.constants.Status.INSTANTIATING_COMPONENTS, total=len(component_configs)
            )

            components = {}
            for i, (key, config_path, display_name) in enumerate(component_configs):
                progress.update(
                    task,
                    completed   = i,
                    description = self.constants.Status.SETUP_COMPONENT_TPL.format(
                        display_name=display_name
                    )
                )
                sleep(0.2)  # Brief pause for visual effect

                if "." in config_path:
                    parent, child = config_path.split(".")
                    config_obj = getattr(cfg, parent)[child]
                else:
                    config_obj = getattr(cfg, config_path)

                components[key] = instantiate(config_obj, _parser=pydantic_parser)

            progress.update(task, completed=len(component_configs))

            visualizer_key = self.constants.Training.VISUALIZER_KEY
            if hasattr(cfg, visualizer_key):
                components['visualizer'] = instantiate(
                    getattr(cfg, visualizer_key),
                    _parser=pydantic_parser
                )
            else:
                components['visualizer'] = None

        return components

    def _perform_system_validation(self):
        """
        Performs comprehensive system validation checks.

        This helper validates hardware capabilities, software versions, and
        integration status before proceeding with training initialization. It
        includes a brief pause to improve the user's perception of the check.
        """
        self.ui.print_section(self.constants.Sections.SYSTEM_VALIDATION, "thermal")

        with self.ui.console.status(
            self.constants.Status.CHECKING_REQS,
            spinner="dots"
        ):
            sleep(0.5)  # Brief pause for visual effect
            info = self.system.get_system_info(self.constants)
            table = self.ui.create_system_table(info)

        self.ui.console.print(table)
        self.ui.console.print()

        status, details = self.system.check_wandb_status(self.constants)
        self.ui.console.print(f"[swarm]📊 wandb: {status} • {details}[/swarm]")
        self.ui.console.print()

    def _run_training(
        self,
        config_overrides : list[str] | None,
        preset           : str | None,
        wandb_project    : str,
    ):
        """
        Executes the training workflow with progress tracking.

        This function manages the lazy loading of training dependencies and
        orchestrates the component initialization process with visual feedback,
        finally calling the Hydra-decorated function.

        Args:
            config_overrides : List of Hydra configuration overrides.
            preset           : The configuration preset to use.
            wandb_project    : The wandb project name.
        """
        self.ui.print_message(self.constants.Messages.LOADING_COMPONENTS, "info")

        hydra_train = self._build_hydra_configuration(
            config_overrides = config_overrides,
            preset           = preset,
            wandb_project    = wandb_project,
        )

        self.ui.console.print()

        if preset or config_overrides:
            args = []
            if preset:
                args.append(f"+preset={preset}")
            if config_overrides:
                args.extend(config_overrides)
            argv.extend(args)

        hydra_train()

    def _setup_configuration(
        self,
        config_overrides : list[str] | None,
        force            : bool,
        interactive      : bool,
        preset           : str | None,
        wandb_project    : str | None,
    ) -> tuple[str | None, str, list[str] | None]:
        """
        Sets up and validates the complete training configuration.

        This method handles preset selection, wandb project naming, and custom
        configuration overrides through either interactive prompts or command-line
        flags, then validates the resulting configuration before proceeding.

        Args:
            config_overrides : A list of Hydra configuration overrides.
            force            : If True, skips system validation checks.
            interactive      : If True, enables interactive prompts for configuration.
            preset           : The name of the configuration preset to use.
            wandb_project    : The name of the wandb project for tracking.

        Returns:
            A tuple containing the final (preset, wandb_project, config_overrides).
        """
        self.ui.print_section(self.constants.Sections.CONFIG_SETUP, "accent")

        if interactive and not preset:
            preset = self.prompts.select_configuration_preset()

        if preset:
            self.ui.print_message(
                f"Using preset: [bright_cyan]{preset}[/bright_cyan]",
                "config"
            )

        if interactive and not wandb_project:
            wandb_project = self.prompts.ask_wandb_project_name()
        elif not wandb_project:
            wandb_project = self.constants.Wandb.DEFAULT_PROJECT

        if interactive and not config_overrides:
            additional       = self.prompts.ask_for_config_overrides()
            config_overrides = (config_overrides or []) + additional

        issues = self.system.validate_config_overrides(config_overrides, self.constants)

        if issues and not force:
            self._handle_configuration_issues(interactive, issues)

        return preset, wandb_project, config_overrides
