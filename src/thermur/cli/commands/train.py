"""
Training command for the Thermur CLI.

This module encapsulates all logic for the 'train' command, including
system validation, configuration, and the initialization of the
imitation learning workflow.
"""
from omegaconf import OmegaConf
from textwrap  import shorten
from typer     import Context, Exit, Option


def train(
    ctx              : Context,
    config_overrides : list[str] | None = Option(
        None,
        "--config", "-c",
        help = "Hydra configuration overrides"
    ),
    dry_run          : bool = Option(
        False,
        "--dry-run", "-d",
        help = "Show configuration and setup without training"
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
        thermur train --dry-run                         # Validate config without training
    """
    command = TrainCommand(ctx)
    command.run(
        config_overrides = config_overrides,
        dry_run          = dry_run,
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
        gpu = self.system.get_system_info().get("cuda", False)
        preset_display = (
            self.cfg.presets.presets.get(preset, {}).get('emoji', preset)
            if preset else "🧵"
        )
            
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
        
    def _create_configuration_table(
        self,
        cfg,
        title : str = "Training Configuration"
    ):
        """
        Create a formatted table displaying the training configuration.
        
        This method automatically traverses the entire configuration tree
        and displays all parameters in a hierarchical format.
        
        Args:
            cfg   : The resolved configuration object
            title : Optional title for the table
            
        Returns:
            A Rich Table object with the formatted configuration
        """
        columns = [
            ("Configuration Path", "bright_cyan", 40, "left"),
            ("Value", "bright_white", 35, "left")
        ]
        
        table = self.ui.create_aligned_table(
            border_style = "bright_blue",
            columns      = columns,
            title        = title
        )
        
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        flat_config = self._flatten_config(cfg_dict)
        
        for path, value in sorted(flat_config.items()):
            display_value = str(value)
            if len(display_value) > 35:
                display_value = shorten(display_value, width=35, placeholder="...")
            table.add_row(path, display_value)
        
        return table
    
    def _display_configuration(
        self,
        cfg,
        config_overrides : list[str] | None,
        preset           : str | None
    ):
        """
        Display the full configuration for the training pipeline.
        
        Shows both the resolved configuration values and metadata about
        how the configuration was constructed (preset, overrides, etc.).
        
        Args:
            cfg              : The resolved configuration object
            config_overrides : The list of custom configuration overrides
            preset           : The selected configuration preset
        """
        from rich.panel import Panel
        
        config_table = self._create_configuration_table(cfg)
        self.ui.console.print(config_table)
        
        meta_info = []
        if preset:
            preset_info = self.cfg.presets.presets.get(preset, {})
            emoji = preset_info.get('emoji', '')
            meta_info.append(f"Preset: {emoji} {preset}")
        
        if config_overrides:
            meta_info.append(f"Overrides: {len(config_overrides)}")
            meta_info.extend(f"  • {override}" for override in config_overrides)
        
        if meta_info:
            self.ui.console.print()
            self.ui.console.print(
                Panel(
                    "\n".join(meta_info),
                    title        = "Configuration Source",
                    border_style = "dim"
                )
            )

    def _execute_dry_run(
        self,
        cfg,
        config_overrides : list[str] | None,
        preset           : str | None
    ):
        """
        Execute the dry-run workflow to display configuration without training.
        
        This method shows the resolved configuration and exits gracefully,
        allowing users to verify their setup before actual training.
        
        Args:
            cfg              : The resolved Hydra configuration object.
            config_overrides : The list of custom configuration overrides.
            preset           : The selected configuration preset.
            
        Returns:
            Dictionary with dry-run completion status.
        """
        self.ui.print_message(
            message  = self.cfg.messages.dry_run_header,
            msg_type = "warning"
        )
        self.ui.console.print()
        
        self._display_configuration(
            cfg              = cfg,
            config_overrides = config_overrides,
            preset           = preset
        )
        
        self.ui.console.print()
        self.ui.print_message(
            message  = self.cfg.messages.dry_run_complete,
            msg_type = "success"
        )
        
        return {"status": "dry_run_complete"}
    
    def _execute_training_workflow(self, cfg):
        """
        Executes the main training workflow with instantiated components.

        This function takes the resolved Hydra configuration and uses it to set up
        logging, random seeds, and instantiate all necessary training components
        before starting the main training loop.

        Args:
            cfg: The resolved Hydra configuration object.
        """
        imports                  = self._training_imports
        configure_loguru         = imports["configure_loguru"]
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

    def _flatten_config(self, config, parent_key="", separator="."):
        """
        Flatten nested configuration dictionary into dot-notation paths.
        
        Args:
            config      : The configuration dictionary to flatten
            parent_key  : The parent key path (used for recursion)
            separator   : The separator to use between keys
            
        Returns:
            Dictionary mapping dot-notation paths to values
        """
        items = {}
        
        for key, value in config.items():
            if key.startswith('_'):
                continue
                
            new_key = separator.join(filter(None, [parent_key, key]))
            
            if isinstance(value, dict):
                items.update(self._flatten_config(value, new_key, separator))
            else:
                items[new_key] = value
                
        return items

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
            for i, issue in enumerate(issues, start=1):
                self.ui.console.print(f"  {i}. {issue}")
            self.ui.print_message(
                message  = self.cfg.messages.validation["force_override"],
                msg_type = "info"
            )
            raise Exit(1)
    
    def _initiate_training(
        self,
        config_overrides : list[str] | None,
        dry_run          : bool,
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
            dry_run          : If True, exits after showing configuration.
            preset           : The selected configuration preset.
            wandb_project    : The configured wandb project name.
        """
        self.ui.print_minor_section("Initializing Training")

        self.ui.print_wandb_info(
            project = wandb_project,
            url     = self.system.get_wandb_url(wandb_project)
        )
        self.ui.console.print()

        try:
            self._run_training(
                config_overrides = config_overrides,
                dry_run          = dry_run,
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
            component_cfgs = self.cfg.cli.training_component_configs
            task = progress.add_task(
                description = self.cfg.messages.status["instantiating_components"],
                total       = len(component_cfgs)
            )

            components = {}
            for i, (key, config_path, display_name) in enumerate(component_cfgs):
                progress.update(
                    completed   = i,
                    description = (
                        self.cfg.messages.status["setup_component_template"]
                            .format(display_name=display_name)
                    ),
                    task_id     = task
                )
                config_obj = OmegaConf.select(cfg, config_path)
                
                if config_obj is None:
                    raise ValueError(f"Configuration path '{config_path}' not found")

                components[key] = instantiate(config_obj, pydantic_parser)

            progress.update(
                completed = len(component_cfgs),
                task_id   = task
            )

            if visualizer_cfg := OmegaConf.select(cfg, "visualizer"):
                components['visualizer'] = instantiate(visualizer_cfg, pydantic_parser)
            else:
                components['visualizer'] = None

        return components

    def _perform_system_validation(self):
        """
        Performs comprehensive system validation checks.

        This helper validates hardware capabilities, software versions, and
        integration status before proceeding with training initialization.
        """
        self.ui.print_major_section("System Information")

        with self.ui.console.status(
            spinner = "dots",
            status  = self.cfg.messages.status["checking_reqs"]
        ):
            info = self.system.get_system_info()

        self.ui.console.print(self.ui.create_system_table(info))
        self.ui.console.print()

        status, details = self.system.check_wandb_status()
        self.ui.console.print(f"[flock]🎨 wandb: {status} • {details}[/flock]")
        self.ui.console.print()

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

    def _run_training(
        self,
        config_overrides : list[str] | None,
        dry_run          : bool,
        preset           : str | None
    ):
        """
        Executes the training workflow using hydra_zen.launch.

        This function manages the lazy loading of training dependencies and
        launches the training job using hydra_zen's programmatic interface,
        which is cleaner than manipulating sys.argv.

        Args:
            config_overrides : List of Hydra configuration overrides.
            dry_run          : If True, shows configuration without training.
            preset           : The configuration preset to use.
        """
        self.ui.print_message(
            message  = self.cfg.messages.loading_components,
            msg_type = "info"
        )

        imports = self._prepare_training_imports()
        self.ui.console.print()

        overrides = (
            ([f"+preset={preset}"] if preset in self.cfg.presets.presets else [])
            + (config_overrides or [])
        )

        def training_task(cfg):
            self._training_imports = imports
            self._resolved_cfg     = cfg
            if dry_run:
                return self._execute_dry_run(
                    cfg              = cfg,
                    config_overrides = config_overrides,
                    preset           = preset
                )
            return self._execute_training_workflow(cfg)

        job = imports["launch"](
            imports["imitation_cfg"],
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

        if interactive and not preset:
            preset = self.prompts.select_configuration_preset()

        match (preset, interactive):
            case (preset, _) if preset:
                preset_emoji = self.cfg.presets.presets.get(preset, {}).get('emoji', preset)
                self.ui.print_message(
                    message  = f"Using preset: [bright_cyan]{preset_emoji}[/bright_cyan]",
                    msg_type = "config"
                )
            case (None, False):
                self.ui.print_message(
                    message  = "Using default configuration",
                    msg_type = "config"
                )

        default_project = self.cfg.wandb_integration.default_project
        wandb_project = (
            self.prompts.ask_wandb_project_name(default_project)
            if interactive and not wandb_project
            else wandb_project or default_project
        )

        if interactive and not config_overrides:
            additional       = self.prompts.ask_for_config_overrides()
            config_overrides = (config_overrides or []) + additional

        if not force:
            issues = self.system.validate_config_overrides(
                overrides = config_overrides
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
        dry_run          : bool,
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
            dry_run          : If True, shows configuration without training.
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
            dry_run          = dry_run,
            preset           = preset,
            wandb_project    = wandb_project
        )
