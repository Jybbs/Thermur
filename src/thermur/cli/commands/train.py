"""
Training command for the Thermur CLI.

This module encapsulates all logic for the 'train' command, including
system validation, configuration, and the initialization of the
imitation learning workflow.
"""
from omegaconf import OmegaConf
from typer     import Context, Exit, Option


def train(
    ctx              : Context,
    config_overrides : list[str] | None = Option(
        None,
        "--config", "-c",
        help = "Hydra configuration overrides"
    ),
    force            : bool = Option(
        False,
        "--force", "-f",
        help = "Skip system checks and warnings"
    ),
    interactive      : bool = Option(
        True,
        "--interactive/--no-interactive", "-i/-n",
        help = "Enable interactive configuration prompts"
    ),
    preset           : str | None = Option(
        None,
        "--preset", "-p",
        help = "Configuration preset (quick, standard, large, debug)"
    ),
    wandb_project    : str | None = Option(
        None,
        "--wandb-project", "-w",
        help = "wandb project name for experiment tracking"
    )
):
    """
    🚀 Train the thermal drone flock using imitation learning.

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
        wandb_project    = wandb_project
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
        self.cfg     = ctx.obj.cfg
        self.prompts = ctx.obj.prompts
        self.system  = ctx.obj.system
        self.ui      = ctx.obj.ui

    def _prepare_training_imports(self):
        """
        Lazily imports heavy dependencies for training.

        This method imports PyTorch, Hydra, and other training dependencies
        only when training is actually initiated, ensuring that other CLI
        commands have a fast startup time.

        Returns:
            A dictionary of imported modules and functions.
        """
        with self.ui.create_thermal_progress() as progress:
            task = progress.add_task(
                description = self.cfg.messages.status["init_modules"],
                total       = 100
            )

            progress.update(
                advance     = 20,
                description = self.cfg.messages.status["loading_config_sys"],
                task_id     = task
            )
            from configs.imitation              import imitation_cfg, register_imitation_cfgs
            from hydra_zen                      import instantiate, launch
            from hydra_zen.third_party.pydantic import pydantic_parser
            from thermur.utils                  import configure_loguru
            from thermur.utils                  import set_seed
            from thermur.training               import train_imitation_learning

            progress.update(
                advance     = 30,
                description = self.cfg.messages.status["registering_configs"],
                task_id     = task
            )
            register_imitation_cfgs()

            progress.update(
                advance     = 50,
                description = self.cfg.messages.status["preparing_hydra"],
                task_id     = task
            )

            imports = {
                "configure_loguru"         : configure_loguru,
                "imitation_cfg"            : imitation_cfg,
                "instantiate"              : instantiate,
                "launch"                   : launch,
                "pydantic_parser"          : pydantic_parser,
                "set_seed"                 : set_seed,
                "train_imitation_learning" : train_imitation_learning
            }

            progress.update(
                completed   = 100,
                description = self.cfg.messages.status["ready_to_train"],
                task_id     = task
            )

        return imports

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
        gpu = self.system.get_system_info(self.cfg.wandb_integration)["cuda"]
        if preset:
            preset_display = (
                self.cfg.presets.presets
                    .get(preset, {})
                    .get('emoji', preset)
            )
        else:
            preset_display = "🧵"  # Custom preset emoji
            
        if not self.prompts.show_training_summary(
            {
                "gpu_available" : gpu,
                "overrides"     : len(config_overrides or []),
                "preset"        : preset_display,
                "wandb_project" : wandb_project,
            }
        ):
            self.ui.print_message(
                message  = self.cfg.messages.training_cancelled,
                msg_type = "warning"
            )
            raise Exit()

    def _execute_training_workflow(self, cfg):
        """
        Executes the main training workflow with instantiated components.

        This function takes the resolved Hydra configuration and uses it to set up
        logging, random seeds, and instantiate all necessary training components
        before starting the main training loop.

        Args:
            cfg: The resolved Hydra configuration object.
        """
        configure_loguru         = imports["configure_loguru"]
        imports                  = self._training_imports
        instantiate              = imports["instantiate"]
        pydantic_parser          = imports["pydantic_parser"]
        set_seed                 = imports["set_seed"]
        train_imitation_learning = imports["train_imitation_learning"]

        self.ui.print_major_section("Preparing Training Environment")
        self.ui.console.print()

        configure_loguru(cfg.logging)
        set_seed(cfg.learning.seed)
        self.ui.console.print()

        components = self._instantiate_training_components(
            cfg             = cfg,
            instantiate     = instantiate,
            pydantic_parser = pydantic_parser,
        )
        
        components["learning"] = cfg.learning
        components["wandb"]    = cfg.wandb

        self.ui.print_message(
            message  = self.cfg.messages.components_initialized,
            msg_type = "success"
        )
        self.ui.console.print()

        self.ui.print_major_section("Training Started")
        self.ui.print_message(
            message  = self.cfg.messages.monitoring_dynamics,
            msg_type = "thermal"
        )
        self.ui.print_message(
            message  = self.cfg.messages.track_wandb,
            msg_type = "flock"
        )
        self.ui.console.print()

        train_imitation_learning(**components)

        self.ui.console.print()
        self.ui.print_header(title = "Training Complete 🎉")

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
        if interactive:
            if not self.prompts.confirm_system_override(issues):
                self.ui.print_message(
                    message  = self.cfg.messages.training_cancelled,
                    msg_type = "warning"
                )
                raise Exit()
        else:
            self.ui.print_message(
                message  = self.cfg.messages.validation["config_fail"],
                msg_type = "error"
            )
            for issue in issues:
                self.ui.console.print(f"  • {issue}")
            self.ui.print_message(
                message  = self.cfg.messages.validation["force_override"],
                msg_type = "info"
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
        self.ui.print_minor_section("Initializing Training")

        self.ui.print_wandb_info(
            project = wandb_project, 
            url     = self.system.get_wandb_url(
                project           = wandb_project,
                wandb_integration = self.cfg.wandb_integration
            )
        )
        self.ui.console.print()

        try:
            self._run_training(
                config_overrides = config_overrides,
                preset           = preset
            )
        except KeyboardInterrupt:
            self.ui.console.print()
            self.ui.print_message(
                message  = self.cfg.messages.training_interrupted,
                msg_type = "warning"
            )
            raise Exit()
        except Exception as e:
            self.ui.print_message(
                message  = self.cfg.messages.training_failed_template.format(e=e),
                msg_type = "error"
            )
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
            component_cfgs = self.cfg.cli.training_component_cfgs
            task = progress.add_task(
                description = self.cfg.messages.status["instantiating_components"],
                total       = len(component_cfgs)
            )

            components = {}
            for i, (key, config_path, display_name) in enumerate(component_cfgs):
                progress.update(
                    task_id     = task,
                    completed   = i,
                    description = (
                        self.cfg.messages.status["setup_component_template"]
                            .format(display_name=display_name)
                    )
                )
                config_obj = OmegaConf.select(cfg, config_path)
                
                if config_obj is None:
                    raise ValueError(f"Configuration path '{config_path}' not found")

                components[key] = instantiate(config_obj, pydantic_parser)

            progress.update(
                task_id   = task, 
                completed = len(component_cfgs)
            )

            # Handle optional visualizer
            visualizer_cfg = OmegaConf.select(cfg, "visualizer")
            components['visualizer'] = (
                instantiate(visualizer_cfg, pydantic_parser) 
                if visualizer_cfg is not None 
                else None
            )

        return components

    def _perform_system_validation(self):
        """
        Performs comprehensive system validation checks.

        This helper validates hardware capabilities, software versions, and
        integration status before proceeding with training initialization.
        """
        self.ui.print_major_section("System Information")

        with self.ui.console.status(
            status  = self.cfg.messages.status["checking_reqs"],
            spinner = "dots"
        ):
            info = self.system.get_system_info(self.cfg.wandb_integration)

        self.ui.console.print(self.ui.create_system_table(info))
        self.ui.console.print()

        status, details = self.system.check_wandb_status(self.cfg)
        self.ui.console.print(f"[flock]🎨 wandb: {status} • {details}[/flock]")
        self.ui.console.print()

    def _run_training(
        self,
        config_overrides : list[str] | None,
        preset           : str | None
    ):
        """
        Executes the training workflow using hydra_zen.launch.

        This function manages the lazy loading of training dependencies and
        launches the training job using hydra_zen's programmatic interface,
        which is cleaner than manipulating sys.argv.

        Args:
            config_overrides : List of Hydra configuration overrides.
            preset           : The configuration preset to use.
            wandb_project    : The wandb project name.
        """
        self.ui.print_message(
            message  = self.cfg.messages.loading_components,
            msg_type = "info"
        )

        imports = self._prepare_training_imports()
        self.ui.console.print()

        overrides = []
        if preset in list(self.cfg.presets.presets.keys()):
            overrides.append(f"+preset={preset}")
        if config_overrides:
            overrides.extend(config_overrides)

        def training_task(cfg):
            self._training_imports = imports
            return self._execute_training_workflow(cfg)

        job = imports["launch"](
            cfg                    = imports["imitation_cfg"],
            config_name            = "train",
            overrides              = overrides,
            task_function          = training_task,
            version_base           = None,
            with_log_configuration = False,
        )

        if job.status.name != "COMPLETED":
            raise RuntimeError(f"Training job failed with status: {job.status}")

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
        self.ui.print_major_section("Configuration Setup")

        preset = (
            self.prompts.select_configuration_preset() 
            if interactive and not preset 
            else preset
        )

        if preset:
            preset_emoji = (
                self.cfg.presets.presets
                    .get(preset, {})
                    .get('emoji', preset)
            )
            self.ui.print_message(
                message  = f"Using preset: [bright_cyan]{preset_emoji}[/bright_cyan]",
                msg_type = "config"
            )
        elif not interactive:
            self.ui.print_message(
                message  = "Using default configuration",
                msg_type = "config"
            )

        default_project = self.cfg.wandb_integration.default_project
        wandb_project   = (
            self.prompts.ask_wandb_project_name(default_project) 
            if interactive and not wandb_project
            else wandb_project or default_project
        )

        if interactive and not config_overrides:
            additional       = self.prompts.ask_for_config_overrides()
            config_overrides = (config_overrides or []) + additional

        if not force:
            issues = self.system.validate_config_overrides(
                messages          = self.cfg.messages,
                overrides         = config_overrides,
                wandb_integration = self.cfg.wandb_integration
            )
            if issues:
                self._handle_configuration_issues(
                    interactive = interactive,
                    issues      = issues
                )

        return preset, wandb_project, config_overrides
    
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
        self.ui.print_header(title = "Thermur Training System")

        if not force:
            self._perform_system_validation()
        else:
            self.ui.print_message(
                message  = self.cfg.messages.skipping_checks,
                msg_type = "warning"
            )

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
