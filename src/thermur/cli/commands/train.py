"""
Training command for the Thermur CLI.

This module encapsulates all logic for the 'train' command, including
system validation, configuration, and the initialization of the
imitation learning workflow.
"""
from functools   import partial
from itertools   import chain
from omegaconf   import DictConfig, OmegaConf, open_dict
from pathlib     import Path
from subprocess  import run as subrun
from thermur.cli import app
from typer       import Argument, Exit, Option
from typing      import Any, Callable, Optional


def train(
    overrides: list[str] = Argument(
        default = None,
        help    = "Hydra configuration overrides (e.g.,optimizer.learning_rate=0.001)"
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
    resume: Optional[Path] = Option(
        None,
        "--resume", "-r",
        help = "Resume from checkpoint. Path to checkpoint or 'last' for most recent."
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
        thermur train                                   # Interactive training
        thermur train optimizer.learning_rate=0.001     # Override config value
        thermur train +model.new_param=42               # Append new config value
        thermur train ++model.force_param=true          # Force add/override
        thermur train --no-interactive --force          # Non-interactive mode
        thermur train --dry-run                         # Validate config without training
        thermur train --resume last                     # Resume from last checkpoint
        thermur train --resume checkpoints/epoch5.ckpt  # Resume from specific checkpoint
    """
    TrainCommand().run(
        dry_run     = dry_run,
        force       = force,
        interactive = interactive,
        sample      = sample,
        overrides   = overrides or [],
        resume      = resume
    )


class TrainCommand:
    """
    Encapsulates the state and logic for the 'train' command.

    This class provides a structured way to manage the training workflow,
    holding shared components from the Typer context and organizing the
    process into a series of well-defined, testable methods.
    """
    def __init__(self):
        """
        Initializes the command with shared context components.
        """
        self.cfg         = app.cfg
        self.prompts     = app.prompts
        self.system      = app.system
        self.ui          = app.ui

        self.data_path   = ""
        self.dry_run     = False
        self.force       = False
        self.interactive = True
        self.output_dir  = None
        self.overrides   = []
        self.resume      = None
        self.sample      = False

    def _display_override_details(self):
        """
        Show configuration override details in a formatted panel.
        
        The panel provides a quick summary of how the configuration was 
        constructed, helping users understand what settings are being applied.
        """
        if not self.overrides:
            return
        
        meta_info = []
        meta_info.append(f"Overrides: {len(self.overrides)}")
        meta_info.extend(f"  • {o}" for o in self.overrides)
        
        self.ui.console.print()
        self.ui.display_panel(
            border_style = "dim",
            content      = "\n".join(meta_info),
            title        = "Configuration Source"
        )

    def _ensure_data_available(self) -> str:
        """
        Ensure training data is available.
        
        Returns:
            Path to data directory.
            
        Raises:
            Exit: If data not found and user declines download.
        """
        try:
            data_path, msg = self.system.resolve_data_path(self.sample)
            msg and self.ui.print_message(msg, "info")
            return data_path
            
        except FileNotFoundError:
            if self.prompts.confirm("No data found. Download sample dataset?"):
                subrun(["thermur", "download", "--sample"])

                return Path(self.cfg.download.sample_data_path).as_posix()
            
            raise Exit("Training requires data. Run 'thermur download -s'")

    def _find_last_checkpoint(self) -> Path | None:
        """
        Find the most recent checkpoint file.
        
        Returns:
            Path to last checkpoint or None if not found.
        """
        outputs = Path("outputs")
        if not outputs.exists():
            return None
            
        checkpoints = [
            run / "checkpoints" / "last.ckpt"
            for run in chain.from_iterable(
                sorted(date.iterdir(), reverse=True)
                for date in sorted(outputs.iterdir(), reverse=True)
                if date.is_dir()
            )
            if run.is_dir() and (run / ".hydra").exists()
        ]
        
        return next((ckpt for ckpt in checkpoints if ckpt.exists()), None)

    def _handle_config_issues(self, issues: list[str]):
        """
        Handles reported configuration validation issues.

        If in interactive mode, it displays the issues and asks the user for
        confirmation to proceed. In non-interactive mode, it prints the errors
        and exits the application with a non-zero status code.

        Args:
            issues : A list of validation issue strings to display.
        """
        if self.interactive:
            if not self.prompts.confirm_system_override(issues):
                self.ui.print_message(
                    message  = "Training cancelled by user.",
                    msg_type = "warning"
                )
                raise Exit()
        else:
            self.ui.print_message(
                message  = "Configuration validation failed:",
                msg_type = "error"
            )

            for i, issue in enumerate(issues, start=1):
                self.ui.console.print(f"  {i}. {issue}")

            self.ui.print_message(
                message  = "Use --force to override or fix the issues above.",
                msg_type = "info"
            )
            raise Exit(1)

    def _instantiate_components(
        self,
        cfg             : DictConfig,
        instantiate     : Callable,
        pydantic_parser : Callable
    ) -> dict[str, Any]:
        """
        Create concrete objects from the configuration.
        
        Components are instantiated in the order specified by the 
        training_component_configs, with a progress bar showing
        the instantiation process.

        Args:
            cfg             : Resolved hydra configuration.
            instantiate     : Hydra-zen instantiate function.
            pydantic_parser : Parser for pydantic models.

        Returns:
            Dictionary of instantiated components ready for training.
            
        Raises:
            ValueError: If configuration path is not found.
        """
        with self.ui.create_thermal_progress() as progress:
            component_cfgs = self.cfg.display.training_component_configs
            task = progress.add_task(
                description = "Instantiating components...",
                total       = len(component_cfgs)
            )

            components = {}
            for i, (key, path, display_name) in enumerate(component_cfgs):
                progress.update(
                    completed   = i,
                    description = f"Setting up {display_name}...",
                    task_id     = task
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
                if (c := OmegaConf.select(cfg, "_system.visualizer"))
                else None
            )

        return components

    def _launch_hydra(self):
        """
        Executes training via hydra-zen launch mechanism.

        This method handles both dry-run and actual training modes, automatically
        selecting the appropriate task function.
            
        Raises:
            RuntimeError: If the hydra job fails to complete successfully.
        """
        self.ui.print_message(
            message  = "Loading training components...",
            msg_type = "info"
        )

        imports = self._load_training_modules()
        self.ui.console.print()

        task_function = (
            self._task if self.dry_run 
            else partial(self._task, imports=imports)
        )

        job = imports["launch"](
            imports["ImitationConfig"],
            overrides              = self.overrides,
            task_function          = task_function,
            version_base           = None,
            with_log_configuration = False,
        )

        if job.status.name != "COMPLETED":
            raise RuntimeError(f"Training job failed with status: {job.status}")

    def _load_training_modules(self) -> dict[str, Callable]:
        """
        Lazily imports heavy dependencies for training to keep the CLI lean.

        Returns:
            A dictionary of imported modules and functions.
        """
        with self.ui.create_thermal_progress() as progress:
            task = progress.add_task(
                description = "Initializing core modules...",
                total       = 100
            )

            progress.update(
                advance     = 20,
                description = "Loading configuration system...",
                task_id     = task
            )

            from config.imitation               import ImitationConfig
            from hydra_zen                      import instantiate, launch
            from hydra_zen.third_party.pydantic import pydantic_parser
            from pytorch_lightning              import seed_everything

            progress.update(
                advance     = 30,
                description = "Registering configurations...",
                task_id     = task
            )

            progress.update(
                advance     = 50,
                description = "Preparing Hydra runtime...",
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
                description = "Ready to train!",
                task_id     = task
            )

        return imports

    def _offer_config_viewing(self):
        """
        Offers to view configuration via the runs command.
        """
        relative_path = self.output_dir.relative_to(Path.cwd())
        
        self.ui.console.print()
        self.ui.print_message(
            message  = (
                f"View configuration with: "
                f"[bold]thermur runs show {relative_path}[/bold]"
            ),
            msg_type = "info"
        )
        
        should_view = (
            not self.interactive or 
            self.prompts.confirm("Would you like to view the configuration now?")
        )
        
        if should_view:
            subrun(['thermur', 'runs', 'show', str(relative_path)])

    def _request_confirmation(self):
        """
        Presents a final summary of the training configuration, including GPU 
        availability and number of overrides.
        
        It acts as a final checkpoint before launching the potentially long-running 
        training process.
            
        Raises:
            Exit: If the user declines to proceed with training.
        """
        system_info = self.system.get_system_info()
        
        summary_data = {
            "gpu_available" : system_info.get("cuda", False),
            "overrides"     : len(self.overrides),
            "wandb_project" : self.cfg.wandb.project,
        }
            
        if not self.prompts.show_training_summary(summary_data):
            self.ui.print_message(
                message  = "Training cancelled by user.",
                msg_type = "warning"
            )
            raise Exit()

    def _task(
        self, 
        cfg     : DictConfig          = None, 
        imports : dict[str, Callable] = None
    ):
        """
        Execute the training or dry-run workflow.

        In dry-run mode, shows configuration without executing training.
        In normal mode, sets up training environment and runs the training loop.

        Args:
            cfg     : Resolved hydra configuration (passed by Hydra if needed).
            imports : Lazy-loaded training modules (None in dry-run mode).

        Returns:
            Status dictionary indicating completion.
        """
        self.output_dir = self.system.get_hydra_output_dir()
        
        if self.dry_run:
            self.ui.print_message(
                message  = (
                    "[bold yellow]DRY RUN MODE[/bold yellow] - "
                    "No training will occur"
                ),
                msg_type = "warning"
            )
            self.ui.console.print()
            
            self._display_override_details()
            
            self.ui.console.print()
            self.ui.print_message(
                message  = "Dry run complete. Configuration validated successfully.",
                msg_type = "success"
            )
            
            self.system.create_status_marker("dry_run")
            self._offer_config_viewing()
            
            return {
                "output_dir" : str(self.output_dir),
                "status"     : "dry_run_complete"
            }
        
        self.ui.print_message(
            message  = f"Training output: {self.output_dir}",
            msg_type = "info"
        )
        self.ui.console.print()
        
        self.ui.print_section("Preparing Training Environment")
        self.ui.console.print()

        if cfg:
            with open_dict(cfg):
                cfg.simulation.loader.wrf.data_path = self.data_path

            if cfg.optimizer.seed is not None:
                imports["seed_everything"](cfg.optimizer.seed)
        
        self.ui.console.print()

        components = self._instantiate_components(
            cfg             = cfg,
            instantiate     = imports["instantiate"],
            pydantic_parser = imports["pydantic_parser"],
        )

        self.ui.print_message(
            message  = "All components initialized successfully!",
            msg_type = "success"
        )
        self.ui.console.print()

        self.ui.print_section("Training Started")
        if self.resume:
            self.ui.print_message(
                message  = f"Resuming from checkpoint: {self.resume}",
                msg_type = "info"
            )
        self.ui.print_message(
            message  = "Monitoring thermal constraints and flock dynamics",
            msg_type = "thermal"
        )
        self.ui.print_message(
            message  = "Track progress in your wandb dashboard",
            msg_type = "flock"
        )
        self.ui.console.print()

        components["trainer"].fit(
            ckpt_path  = str(self.resume) if self.resume else None,
            datamodule = components["datamodule"],
            model      = components["policy"]
        )

        self.ui.console.print()
        self.ui.print_header("Training Complete 🎉")
        
        self.system.create_status_marker("training_complete")
        self._offer_config_viewing()
        
        return {
            "output_dir" : str(self.output_dir),
            "status"     : "training_complete"
        }

    def run(
        self,
        dry_run     : bool,
        force       : bool,
        interactive : bool,
        sample      : bool,
        overrides   : list[str] | None,
        resume      : Optional[Path],
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
            sample      : If True, use sample data.
            overrides   : A list of Hydra configuration overrides.
            resume      : Optional checkpoint path to resume training from.
        """
        # Set command flags as instance attributes
        self.dry_run     = dry_run
        self.force       = force
        self.interactive = interactive
        self.sample      = sample
        
        if resume:
            if str(resume) == "last":
                self.resume = self._find_last_checkpoint()
                if not self.resume:
                    self.ui.print_message(
                        "No checkpoint found to resume from", "error"
                    )
                    raise Exit(1)
            elif not resume.exists():
                self.ui.print_message(
                    f"Checkpoint not found: {resume}", "error"
                )
                raise Exit(1)
            else:
                self.resume = resume
        else:
            self.resume = None
                
        if overrides:
            self.overrides = overrides
        
        self.ui.print_header("Thermur Training System")

        if not self.force:
            self.ui.display_system_validation(self.system)
        else:
            self.ui.print_message(
                message  = "Skipping system checks (--force enabled)",
                msg_type = "warning"
            )
        
        # Gather additional overrides in interactive mode
        if self.interactive:
            self.overrides = self.overrides + self.prompts.ask_for_overrides()

        self.data_path = self._ensure_data_available()

        if not self.force and (
            issues := self.system.validate_overrides(self.overrides)
        ):
            self._handle_config_issues(issues)

        if self.interactive:
            self._request_confirmation()

        self.ui.print_section("Initializing Training", True)
        if self.cfg.wandb.mode != "disabled":
            self.ui.display_wandb("train", self.cfg.wandb.project)
        self.ui.console.print()

        try:
            self._launch_hydra()

        except KeyboardInterrupt:
            self.ui.console.print()
            self.ui.print_message(
                message  = "Training interrupted by user.",
                msg_type = "warning"
            )
            raise Exit()
        
        except Exception as e:
            try:
                from hydra.errors import ConfigCompositionException
                from hydra.errors import InstantiationException
                from hydra.errors import OverrideParseException
                from pydantic     import ValidationError
            except ImportError:
                self.ui.print_message(f"Training failed: {e}", "error")
                raise Exit(1)
            
            match e:
                case OverrideParseException():
                    self.ui.print_message("Override syntax error:", "error")
                    self.ui.console.print(f"  {e}")
                    self.ui.console.print()
                    self.ui.print_message(
                        "Syntax: key=value, +key=value (append), ++key=value (force)",
                        "info"
                    )
                
                case ConfigCompositionException():
                    self.ui.print_message("Configuration error:", "error")
                    self.ui.console.print(f"  {e}")
                    if hasattr(e, 'available_options'):
                        self.ui.console.print()
                        self.ui.print_message(
                            message  = (
                                f"Available options: "
                                f"{', '.join(e.available_options)}"
                            ),
                            msg_type = "info"
                        )
                
                case InstantiationException():
                    self.ui.print_message("Component instantiation failed:", "error")
                    self.ui.console.print(f"  {e}")
                    if (
                        hasattr(e, '__cause__') 
                        and isinstance(e.__cause__, ValidationError)
                    ):
                        self.ui.console.print()
                        self.ui.print_message("Validation errors:", "error")
                        for error in e.__cause__.errors():
                            self.ui.console.print(
                                f"  - {'.'.join(str(x) for x in error['loc'])}: "
                                f"{error['msg']}"
                            )
                
                case ValidationError():
                    self.ui.print_message("Configuration validation failed:", "error")
                    for error in e.errors():
                        self.ui.console.print(
                            f"  - {'.'.join(str(x) for x in error['loc'])}: "
                            f"{error['msg']}"
                        )
                
                case _:
                    self.ui.print_message(f"Training failed: {e}", "error")
            
            raise Exit(1)
