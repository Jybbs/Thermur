"""
Training command for the Thermur CLI.

This module encapsulates all logic for the 'train' command, including
system validation, configuration, and the initialization of the
imitation learning workflow.
"""
from functools import partial
from omegaconf import DictConfig, OmegaConf, open_dict
from pathlib   import Path
from textwrap  import shorten
from typer     import Argument, Context, Exit, Option
from typing    import Any

import subprocess


def train(
    ctx: Context,
    overrides: list[str] = Argument(
        default = None,
        help    = "Hydra configuration overrides (e.g., learning.lr=0.01 flock.num_drones=20)"
    ),
    dry_run: bool = Option(
        False,
        "--dry-run", "-d",
        help = "Show configuration and setup without training"
    ),
    force: bool = Option(
        False,
        "--force", "-f",
        help = "Skip system checks and warnings"
    ),
    interactive: bool = Option(
        True,
        "--interactive/--no-interactive", "-i/-n",
        help = "Enable interactive configuration prompts"
    ),
    resume: Path | None = Option(
        None,
        "--resume", "-r",
        help = "Resume training from checkpoint file"
    ),
    sample: bool = Option(
        False,
        "--sample", "-s",
        help = "Use bundled sample data instead of downloaded files"
    )
):
    """
    🚀 Train the thermal drone flock using imitation learning.

    This command provides a comprehensive training workflow with system
    validation, configuration management, and seamless wandb integration.

    Examples:
        thermur train                                # Interactive training
        thermur train optimizer.learning_rate=0.001  # Custom parameters
        thermur train --no-interactive --force       # Non-interactive mode
        thermur train --dry-run                      # Validate config without training
        thermur train --resume checkpoints/last.ckpt # Resume from checkpoint
    """
    overrides = overrides or []
    command   = TrainCommand(ctx)
    command.run(
        dry_run     = dry_run,
        force       = force,
        interactive = interactive,
        overrides   = overrides,
        resume      = resume,
        sample      = sample
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

    def _build_overrides(
        self,
        additional : list[str],
        base       : list[str] | None,
    ) -> list[str]:
        """
        Constructs the final override list in the proper order:
        1. Command-line overrides
        2. Interactive overrides

        The ordering ensures that later overrides can supersede earlier ones,
        giving users full control over the final configuration.

        Args:
            additional : Overrides from interactive prompts.
            base       : Command-line provided overrides.

        Returns:
            Complete list of hydra overrides.
        """
        return [
            *(base or []),
            *additional
        ]

    def _create_config_table(
        self, 
        cfg   : DictConfig, 
        title : str = "Training Configuration"
    ):
        """
        Traverses the entire configuration tree and displays all parameters in
        a hierarchical format.
        
        Args:
            cfg   : The resolved Hydra configuration object.
            title : Optional title for the table.
            
        Returns:
            A Rich Table object with the formatted configuration.
        """
        columns = [
            ("Configuration Path", "bright_cyan",  40, "left"),
            ("Value",              "bright_white", 35, "left")
        ]
        
        table = self.ui.create_aligned_table(
            border_style = "bright_blue",
            columns      = columns,
            title        = title
        )
        
        flat_config = self._flatten_config(
            OmegaConf.to_container(cfg, resolve=True)
        )
        
        for path, value in sorted(flat_config.items()):
            display = str(value)
            formatted = (
                shorten(display, width=35, placeholder="...") 
                if len(display) > 35 
                else display
            )
            table.add_row(path, formatted)
        
        return table

    def _display_override_details(self, overrides: list[str]):
        """
        Shows configuration override details in a formatted panel.
        
        The panel provides a quick summary of how the configuration was constructed, 
        helping users understand what settings are being applied.
        
        Args:
            overrides : List of hydra configuration overrides.
        """
        if not overrides:
            return
        
        meta_info = []
        meta_info.append(f"Overrides: {len(overrides)}")
        meta_info.extend(f"  • {o}" for o in overrides)
        
        self.ui.console.print()
        self.ui.display_panel(
            border_style = "dim",
            content      = "\n".join(meta_info),
            title        = "Configuration Source"
        )


    def _dry_run_task(self, cfg: DictConfig, overrides: list[str]):
        """
        Shows the fully-rsolved configuration without executing training.
        
        Alows users to verify their settings before committing to a potentially
        long-running training process.
        
        Args:
            cfg       : Resolved hydra configuration.
            overrides : Configuration overrides for display.
            
        Returns:
            Status dictionary indicating dry run completion.
        """
        self.ui.print_message(
            message  = self.cfg.messages.dry_run_header,
            msg_type = "warning"
        )
        self.ui.console.print()
        
        table = self._create_config_table(cfg)
        self.ui.console.print(table)
        
        self._display_override_details(overrides)
        
        self.ui.console.print()
        self.ui.print_message(
            message  = self.cfg.messages.dry_run_complete,
            msg_type = "success"
        )
        
        return {"status": "dry_run_complete"}

    def _ensure_data_available(self, use_sample: bool) -> str:
        """
        Ensures training data is available.
        
        Args:
            use_sample: Use sample data if True.
            
        Returns:
            Path to data directory.
        """
        try:
            data_path, msg = self.system.resolve_data_path(use_sample=use_sample)
            msg and self.ui.print_message(msg, "info")
            return data_path
            
        except FileNotFoundError:
            if self.prompts.confirm("No data found. Download sample dataset?"):
                subprocess.run(["thermur", "download", "--sample"])

                return Path(self.cfg.download.sample_data_path).as_posix()
            
            raise Exit("Training requires data. Run 'thermur download -s'")

    def _flatten_config(self, config, parent_key="", separator="."):
        """
        Flattens nested configuration dictionary into dot-notation paths.
        
        Private keys (starting with underscore) are filtered out. This flattened 
        representation makes it easy to display configuration in a tabular 
        format for the CLI.
        
        Example:
            {"model": {"layers": 3}} -> {"model.layers": 3}
        
        Args:
            config      : The configuration dictionary to flatten.
            parent_key  : The parent key path (used for recursion).
            separator   : The separator to use between keys.
            
        Returns:
            Dictionary mapping dot-notation paths to values.
        """
        items = {}
        
        for key, value in config.items():
            if key.startswith('_'):
                continue
                
            new_key = separator.join(filter(None, [parent_key, key]))
            
            items |= (
                self._flatten_config(value, new_key, separator) 
                     if isinstance(value, dict) 
                     else {new_key: value}
            )
                
        return items

    def _gather_interactive_inputs(self) -> list[str]:
        """
        Guides the user through interactive configuration by prompting for
        any additional Hydra configuration overrides.
        
        The prompts are designed to provide sensible defaults while giving
        users full control over their training configuration.
        
        Returns:
            List of additional overrides.
        """
        self.ui.print_section("Configuration Setup")
        
        return self.prompts.ask_for_overrides()

    def _handle_config_issues(self, interactive : bool, issues : list[str]):
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

    def _instantiate_components(
        self,
        cfg             : DictConfig,
        instantiate,
        pydantic_parser
    ):
        """
        Uses hydra-zen's instantiate function to create concrete objects from the
        configuration. 
        
        Components are instantiated in the order specified by the 
        training_component_configs, with a progress bar showing
        the instantiation process.

        Args:
            cfg             : Resolved hydra configuration.
            instantiate     : Hydra-zen instantiate function.
            pydantic_parser : Parser for pydantic models.

        Returns:
            Dictionary of instantiated components ready for training.
        """
        with self.ui.create_thermal_progress() as progress:
            component_cfgs = self.cfg.display.training_component_configs
            task = progress.add_task(
                description = self.cfg.messages.status["instantiating_components"],
                total       = len(component_cfgs)
            )

            components = {}
            for i, (key, path, display_name) in enumerate(component_cfgs):
                progress.update(
                    completed   = i,
                    description = (
                        self.cfg.messages.status["setup_component_template"]
                            .format(display_name=display_name)
                    ),
                    task_id = task
                )
                
                if (obj := OmegaConf.select(cfg, path)) is None:
                    raise ValueError(
                        f"Configuration path '{path}' not found"
                    )

                components[key] = instantiate(obj, pydantic_parser)

            progress.update(
                completed = len(component_cfgs),
                task_id   = task
            )

            components['visualizer'] = (
                instantiate(c, pydantic_parser)
                if (c := OmegaConf.select(cfg, "visualizer"))
                else None
            )

        return components

    def _launch_hydra(
        self,
        data_path : str,
        dry_run   : bool,
        overrides : list[str],
        resume    : Path | None,
    ):
        """
        Executes training via hydra-zen launch mechanism.

        This method handles both dry-run and actual training modes, automatically
        selecting the appropriate task function.

        Args:
            dry_run   : Show configuration without training.
            overrides : Complete list of hydra overrides.
            
        Raises:
            RuntimeError: If the hydra job fails to complete successfully.
        """
        self.ui.print_message(
            message  = self.cfg.messages.loading_components,
            msg_type = "info"
        )

        imports = self._load_training_modules()
        self.ui.console.print()

        task_map = {
            True  : partial(self._dry_run_task, overrides=overrides),
            False : partial(
                self._training_task,
                data_path = data_path, 
                imports   = imports, 
                resume    = resume
            )
        }

        job = imports["launch"](
            imports["ImitationConfig"],
            overrides              = overrides,
            task_function          = task_map[dry_run],
            version_base           = None,
            with_log_configuration = False,
        )

        if job.status.name != "COMPLETED":
            raise RuntimeError(f"Training job failed with status: {job.status}")

    def _load_training_modules(self) -> dict[str, Any]:
        """
        Lazily imports heavy dependencies for training to keep the CLI lean.

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

            from config.imitation                      import ImitationConfig
            from hydra_zen                            import instantiate, launch
            from hydra_zen.third_party.pydantic       import pydantic_parser
            from pytorch_lightning                    import seed_everything

            progress.update(
                advance     = 30,
                description = self.cfg.messages.status["registering_configs"],
                task_id     = task
            )

            progress.update(
                advance     = 50,
                description = self.cfg.messages.status["preparing_hydra"],
                task_id     = task
            )

            imports = {
                "ImitationConfig" : ImitationConfig,
                "instantiate"     : instantiate,
                "launch"          : launch,
                "pydantic_parser" : pydantic_parser,
                "seed_everything" : seed_everything
            }

            progress.update(
                completed   = 100,
                description = self.cfg.messages.status["ready_to_train"],
                task_id     = task
            )

        return imports

    def _request_confirmation(
        self,
        overrides     : list[str],
    ):
        """
        Presents a final summary of the training configuration, including GPU 
        availability and number of overrides.
        
        It acts as a final checkpoint before launching the potentially long-running 
        training process.

        Args:
            overrides     : Complete list of hydra overrides.
            
        Raises:
            Exit: If the user declines to proceed with training.
        """
        system_info = self.system.get_system_info()
        
        summary_data = {
            "gpu_available" : system_info.get("cuda", False),
            "overrides"     : len(overrides),
            "wandb_project" : self.cfg.wandb.project,
        }
            
        if not self.prompts.show_training_summary(summary_data):
            self.ui.print_message(
                message  = self.cfg.messages.training_cancelled,
                msg_type = "warning"
            )
            raise Exit()

    def _training_task(
        self, 
        cfg       : DictConfig,
        data_path : str, 
        imports   : dict[str, Any], 
        resume    : Path | None = None
    ):
        """
        Executes the training workflow with resolved configuration.

        It sets up the training environment (logging, random seeds), instantiates 
        all required components from the configuration, and runs the training loop.
        Progress is communicated through UI messages at each major step.

        Args:
            cfg     : Resolved hydra configuration.
            imports : Lazy-loaded training modules.
            resume  : Optional checkpoint path to resume training from.

        Returns:
            Status dictionary indicating successful completion.
        """
        self.ui.print_section("Preparing Training Environment")
        self.ui.console.print()

        with open_dict(cfg):
            cfg.simulation.loader.wrf.data_path = data_path

        if cfg.optimizer.seed is not None:
            imports["seed_everything"](cfg.optimizer.seed)
        self.ui.console.print()

        components = self._instantiate_components(
            cfg             = cfg,
            instantiate     = imports["instantiate"],
            pydantic_parser = imports["pydantic_parser"],
        )

        self.ui.print_message(
            message  = self.cfg.messages.components_initialized,
            msg_type = "success"
        )
        
        self.ui.console.print()

        self.ui.print_section("Training Started")
        if resume:
            self.ui.print_message(
                message  = f"Resuming from checkpoint: {resume}",
                msg_type = "info"
            )
        self.ui.print_message(
            message  = self.cfg.messages.monitoring_dynamics,
            msg_type = "thermal"
        )
        self.ui.print_message(
            message  = self.cfg.messages.track_wandb,
            msg_type = "flock"
        )
        self.ui.console.print()

        components["trainer"].fit(
            model      = components["policy"],
            datamodule = components["datamodule"],
            ckpt_path  = str(resume) if resume else None
        )

        self.ui.console.print()
        self.ui.print_header("Training Complete 🎉")
        
        return {"status": "training_complete"}

    def run(
        self,
        dry_run     : bool,
        force       : bool,
        interactive : bool,
        overrides   : list[str] | None,
        resume      : Path | None,
        sample      : bool,
    ):
        """
        Executes the main training workflow from start to finish.

        This method orchestrates the entire training process, including
        validation, configuration, user confirmation, and the final
        launch of the core training logic.

        Args:
            dry_run     : If True, shows configuration without training.
            force       : If True, skips system validation checks.
            interactive : If True, enables interactive prompts.
            overrides   : A list of Hydra configuration overrides.
            sample      : If True, use sample data.
        """
        self.ui.print_header("Thermur Training System")

        if not force:
            self.ui.display_system_validation(self.system)
        else:
            self.ui.print_message(
                message  = self.cfg.messages.skipping_checks,
                msg_type = "warning"
            )
        
        additional = (
            self._gather_interactive_inputs() if interactive else []
        )

        data_path = self._ensure_data_available(sample)
        overrides = self._build_overrides(
            additional = additional,
            base       = overrides,
        )

        if not force and (issues := self.system.validate_overrides(overrides)):
            self._handle_config_issues(interactive, issues)

        if interactive:
            self._request_confirmation(overrides)

        self.ui.print_section("Initializing Training", minor=True)
        if self.cfg.wandb.mode != "disabled":
            self.ui.display_wandb("train", self.cfg.wandb.project)
        self.ui.console.print()

        try:
            self._launch_hydra(
                data_path = data_path,
                dry_run   = dry_run, 
                overrides = overrides,
                resume    = resume
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