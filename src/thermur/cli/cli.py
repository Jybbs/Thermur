"""
Enhanced console-script target for Thermur, built with Typer and Rich.

This module provides the main CLI interface with visual feedback, system
validation, configuration management, and seamless wandb integration.

The execution flow is:
1. User runs `thermur <command>` in the terminal
2. The `[project.scripts]` entry in `pyproject.toml` points to `src.__main__:cli`
3. `src.__main__.py` calls the `cli` object from this file
4. Typer parses the command-line arguments and invokes the appropriate function
"""
from .ui         import ThermurUI
from .constants  import CLIConstants
from .explorer   import ConfigExplorer
from .prompts    import CLIPrompts
from .system     import SystemInspector
from sys         import argv
from time        import sleep
from typer       import Context, Exit, Option, Typer
from webbrowser  import open

# Create single, shared instances of the core CLI components.
constants = CLIConstants()
ui        = ThermurUI(constants)
system    = SystemInspector()
prompts   = CLIPrompts(ui, constants)


cli = Typer(
    name                     = constants.Core.APP_NAME,
    help                     = constants.Core.APP_DESCRIPTION,
    add_completion           = False,
    rich_markup_mode         = "rich",
    no_args_is_help          = True,
    pretty_exceptions_enable = True,
)


def version_callback(value: bool):
    """
    Show version information with system details.

    Displays comprehensive version information including system hardware,
    software versions, and integration status when the --version flag is used.

    Args:
        value : True if --version flag is present
    """
    if not value:
        return

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


@cli.command("train")
def train(
    ctx: Context,
    preset: str | None = Option(
        None,
        "--preset", "-p",
        help="Configuration preset (quick, standard, large, debug)"
    ),
    interactive: bool = Option(
        True,
        "--interactive/--no-interactive", "-i/-n",
        help="Enable interactive configuration prompts"
    ),
    force: bool = Option(
        False,
        "--force", "-f",
        help="Skip system checks and warnings"
    ),
    wandb_project: str | None = Option(
        None,
        "--wandb-project", "-w",
        help="wandb project name for experiment tracking"
    ),
    config_overrides: list[str] | None = Option(
        None,
        "--config", "-c",
        help="Hydra configuration overrides"
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
    ui.print_header(
        constants.Headers.TRAIN_TITLE,
        constants.Headers.TRAIN_SUBTITLE
    )

    if not force:
        _perform_system_validation()
    else:
        ui.print_message(constants.Messages.SKIPPING_CHECKS, "warning")

    preset, wandb_project, config_overrides = _setup_configuration(
        preset           = preset,
        wandb_project    = wandb_project,
        config_overrides = config_overrides,
        interactive      = interactive,
        force            = force,
    )

    if interactive:
        _confirm_training_setup(preset, wandb_project, config_overrides)

    _initiate_training(preset, config_overrides, wandb_project)


def _perform_system_validation():
    """
    Perform comprehensive system validation checks.

    Validates hardware capabilities, software versions, and integration status
    before proceeding with training initialization. This function includes a
    brief pause to improve the user's perception of the check being performed.
    """
    ui.print_section(constants.Sections.SYSTEM_VALIDATION, "thermal")

    with ui.console.status(
        constants.Status.CHECKING_REQS,
        spinner="dots"
    ):
        sleep(0.5)  # Brief pause for visual effect
        info = system.get_system_info(constants)
        table = ui.create_system_table(info)

    ui.console.print(table)
    ui.console.print()

    status, details = system.check_wandb_status(constants)
    ui.console.print(f"[swarm]📊 wandb: {status} • {details}[/swarm]")
    ui.console.print()


def _setup_configuration(
    preset           : str | None,
    wandb_project    : str | None,
    config_overrides : list[str] | None,
    interactive      : bool,
    force            : bool,
) -> tuple[str | None, str, list[str] | None]:
    """
    Setup and validate training configuration.

    Handles preset selection, wandb project naming, and configuration overrides
    through interactive prompts or command-line flags, then validates the
    resulting configuration before proceeding.

    Args:
        preset           : Configuration preset name
        wandb_project    : wandb project name
        config_overrides : Hydra configuration overrides
        interactive      : Whether to use interactive prompts
        force            : Whether to skip validation

    Returns:
        Tuple of (preset, wandb_project, config_overrides)
    """
    ui.print_section(constants.Sections.CONFIG_SETUP, "accent")

    if interactive and not preset:
        preset = prompts.select_configuration_preset()

    if preset:
        ui.print_message(
            f"Using preset: [bright_cyan]{preset}[/bright_cyan]",
            "config"
        )

    if interactive and not wandb_project:
        wandb_project = prompts.ask_wandb_project_name()
    elif not wandb_project:
        wandb_project = constants.Wandb.DEFAULT_PROJECT

    if interactive and not config_overrides:
        additional       = prompts.ask_for_config_overrides()
        config_overrides = (config_overrides or []) + additional

    issues = system.validate_config_overrides(config_overrides, constants)

    if issues and not force:
        _handle_configuration_issues(issues, interactive)

    return preset, wandb_project, config_overrides


def _handle_configuration_issues(issues: list[str], interactive: bool):
    """
    Handle configuration validation issues.

    If in interactive mode, displays the issues and asks the user to confirm
    if they wish to proceed. In non-interactive mode, it prints the errors
    and exits with a non-zero status code.

    Args:
        issues      : List of validation issues
        interactive : Whether to prompt for confirmation
    """
    if interactive and not prompts.confirm_system_override(issues):
        ui.print_message(constants.Messages.TRAINING_CANCELLED, "warning")
        raise Exit()

    elif not interactive:
        ui.print_message(constants.Validation.CONFIG_FAIL_MSG, "error")

        for issue in issues:
            ui.console.print(f"  • {issue}")
        ui.print_message(
            constants.Validation.FORCE_OVERRIDE_TIP,
            "info"
        )
        raise Exit(1)


def _confirm_training_setup(
    preset           : str | None,
    wandb_project    : str,
    config_overrides : list[str] | None,
):
    """
    Show training summary and get final confirmation.

    Presents a final summary of all settings to the user. This is the last
    chance for a user to cancel before the training process begins.

    Args:
        preset           : Configuration preset
        wandb_project    : wandb project name
        config_overrides : Configuration overrides
    """
    info = system.get_system_info(constants)
    summary = {
        "preset"        : preset or "default",
        "wandb_project" : wandb_project,
        "overrides"     : len(config_overrides or []),
        "gpu_available" : info["cuda"],
    }

    if not prompts.show_training_summary(summary):
        ui.print_message(constants.Messages.TRAINING_CANCELLED, "warning")
        raise Exit()


def _initiate_training(
    preset           : str | None,
    config_overrides : list[str] | None,
    wandb_project    : str,
):
    """
    Initialize and start the training process.

    This function serves as the final gateway before running the core training
    logic, handling any exceptions that may occur during the run.

    Args:
        preset           : Configuration preset
        config_overrides : Configuration overrides
        wandb_project    : wandb project name
    """
    ui.print_section(constants.Sections.INIT_TRAINING, "thermal")

    url = system.get_wandb_url(constants, wandb_project)
    ui.print_wandb_info(wandb_project, url)
    ui.console.print()

    try:
        _run_training(preset, config_overrides, wandb_project)

    except KeyboardInterrupt:
        ui.console.print()
        ui.print_message(constants.Messages.TRAINING_INTERRUPTED, "warning")
        raise Exit()

    except Exception as e:
        ui.print_message(constants.Messages.TRAINING_FAILED_TPL.format(e=e), "error")
        raise Exit(1)


def _run_training(
    preset           : str | None,
    config_overrides : list[str] | None,
    wandb_project    : str
):
    """
    Execute the training workflow with progress tracking.

    This function manages the lazy loading of training dependencies and
    orchestrates the component initialization process with visual feedback.

    Args:
        preset           : Configuration preset to use
        config_overrides : List of Hydra configuration overrides
        wandb_project    : wandb project name
    """
    ui.print_message(constants.Messages.LOADING_COMPONENTS, "info")

    hydra_train = _build_hydra_configuration(
        preset           = preset,
        config_overrides = config_overrides,
        wandb_project    = wandb_project,
    )

    ui.console.print()

    if preset or config_overrides:
        args = []
        if preset:
            args.append(f"+preset={preset}")
        if config_overrides:
            args.extend(config_overrides)
        argv.extend(args)

    hydra_train()


def _build_hydra_configuration(
    preset           : str | None,
    config_overrides : list[str] | None,
    wandb_project    : str,
):
    """
    Build Hydra configuration with lazy imports.

    This function wraps the core training logic in a Hydra-decorated function.
    Heavy dependencies like Hydra and PyTorch are imported lazily inside this
    function to keep the CLI startup fast for other commands.

    Args:
        preset           : Configuration preset
        config_overrides : Configuration overrides
        wandb_project    : wandb project name

    Returns:
        Hydra-decorated training function
    """
    with ui.create_thermal_progress() as progress:
        task = progress.add_task(constants.Status.INIT_MODULES, total=100)

        progress.update(
            task,
            advance     = 20,
            description = constants.Status.LOADING_CONFIG_SYS
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
            description = constants.Status.REGISTERING_CONFIGS
        )
        register_configs()

        progress.update(
            task,
            advance     = 50,
            description = constants.Status.PREPARING_HYDRA
        )

        @zen(imitation_config).hydra_main(
            config_name            = "train",
            config_path            = None,
            version_base           = None,
            with_log_configuration = False,
        )
        def hydra_train(cfg):
            """
            Execute training with resolved configuration.

            This inner function is decorated with Hydra to handle configuration
            resolution and instantiation of all training components.
            """
            _execute_training_workflow(
                cfg                      = cfg,
                instantiate              = instantiate,
                pydantic_parser          = pydantic_parser,
                configure_loguru         = configure_loguru,
                set_seed                 = set_seed,
                train_imitation_learning = train_imitation_learning,
            )

        progress.update(task, advance=100, description=constants.Status.READY_TO_TRAIN)

    return hydra_train


def _execute_training_workflow(
    cfg,
    instantiate,
    pydantic_parser,
    configure_loguru,
    set_seed,
    train_imitation_learning,
):
    """
    Execute the main training workflow with component instantiation.

    This function takes the resolved Hydra configuration and uses it to set up
    logging, random seeds, and instantiate all necessary training components
    before starting the main training loop.

    Args:
        cfg                      : Resolved Hydra configuration
        instantiate              : Hydra-zen instantiate function
        pydantic_parser          : Pydantic parser for instantiation
        configure_loguru         : Loguru configuration function
        set_seed                 : Random seed setting function
        train_imitation_learning : Main training function
    """
    ui.print_section(constants.Sections.BUILDING_COMPONENTS, "thermal")

    configure_loguru(
        instantiate(cfg.monitoring.logging, _parser=pydantic_parser)
    )
    set_seed(
        instantiate(cfg.hyperparameters, _parser=pydantic_parser).seed
    )

    components = _instantiate_training_components(
        cfg             = cfg,
        instantiate     = instantiate,
        pydantic_parser = pydantic_parser,
    )

    ui.print_message(constants.Messages.COMPONENTS_INITIALIZED, "success")
    ui.console.print()

    ui.print_section(constants.Sections.TRAINING_STARTED, "thermal")
    ui.print_message(constants.Messages.MONITORING_DYNAMICS, "thermal")
    ui.print_message(constants.Messages.TRACK_WANDB, "swarm")
    ui.console.print()

    train_imitation_learning(**components)

    ui.console.print()
    ui.print_header(
        constants.Messages.TRAINING_COMPLETE_HEADER,
        constants.Messages.TRAINING_COMPLETE_SUB
    )


def _instantiate_training_components(cfg, instantiate, pydantic_parser):
    """
    Instantiate all training components with progress tracking.

    Iterates through a predefined list of components from the constants file,
    resolves their configurations from the main `cfg` object, and instantiates
    them. A progress bar provides visual feedback on this setup process.

    Args:
        cfg             : Resolved Hydra configuration
        instantiate     : Hydra-zen instantiate function
        pydantic_parser : Pydantic parser

    Returns:
        Dictionary of instantiated components
    """
    with ui.create_thermal_progress() as progress:
        component_configs = constants.Training.COMPONENT_CONFIGS
        task = progress.add_task(
            constants.Status.INSTANTIATING_COMPONENTS, total=len(component_configs)
        )

        components = {}
        for i, (key, config_path, display_name) in enumerate(component_configs):
            progress.update(
                task,
                completed   = i,
                description = constants.Status.SETUP_COMPONENT_TPL.format(
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

        visualizer_key = constants.Training.VISUALIZER_KEY
        if hasattr(cfg, visualizer_key):
            components['visualizer'] = instantiate(
                getattr(cfg, visualizer_key),
                _parser=pydantic_parser
            )
        else:
            components['visualizer'] = None

    return components


@cli.command("configure")
def configure():
    """
    🔧 Interactive configuration explorer and editor.

    Navigate the configuration hierarchy interactively, view Pydantic schemas,
    and edit values without manually writing Hydra overrides.
    """
    explorer  = ConfigExplorer(ui, prompts, constants)
    overrides = explorer.explore_interactive()

    if overrides:
        ui.console.print()
        ui.print_header(constants.Headers.CONFIG_GEN_TITLE)

        ui.print_message(constants.Messages.CONFIG_GEN_ADD_CMD, "info")
        ui.console.print()

        cmd_parts = ["thermur train"]
        for override in overrides:
            cmd_parts.append(f'--config "{override}"')

        command = " ".join(cmd_parts)
        ui.console.print(f"[bold accent]$ {command}[/bold accent]")
        ui.console.print()

        ui.print_message(constants.Messages.CONFIG_GEN_USE_IND, "tip")
        for override in overrides:
            ui.print_config_value("--config", override)
    else:
        ui.print_message(constants.Messages.NO_CONFIG_CHANGES, "info")


@cli.command("info")
def info():
    """
    📊 Display comprehensive system and configuration information.

    Shows detailed information about the current system setup, including
    hardware capabilities, software versions, and configuration status.
    """
    ui.print_header(constants.Headers.INFO_TITLE)

    _perform_system_validation()

    ui.print_section(constants.Sections.FEATURES, "accent")
    features_table = ui.create_feature_table()
    ui.console.print(features_table)

    ui.print_section(constants.Sections.CONFIG_SYSTEM, "config")
    ui.print_config_value(
        "Config Path",
        "configs/",
        "Hydra configuration directory"
    )

    preset_names = ", ".join(constants.Presets.CONFIGS.keys())
    ui.print_config_value(
        "Presets",
        preset_names,
        "Available training presets"
    )

    ui.print_config_value(
        "Explorer",
        "thermur configure",
        "Interactive configuration tool"
    )

    ui.print_training_tips()


@cli.command("validate")
def validate(
    config_overrides: list[str] | None = Option(
        None,
        "--config", "-c",
        help="Configuration overrides to validate"
    )
):
    """
    ✅ Validate system setup and configuration without starting training.

    Performs comprehensive validation of the training environment including
    system requirements, configuration syntax, and integration status.
    """
    ui.print_header(constants.Headers.VALIDATE_TITLE, constants.Headers.VALIDATE_SUBTITLE)

    _perform_system_validation()

    ui.print_section(constants.Sections.CONFIG_CHECK, "config")

    with ui.console.status(
        constants.Status.VALIDATING_CONFIG,
        spinner="dots"
    ):
        sleep(0.3)
        issues = system.validate_config_overrides(config_overrides, constants)

    if issues:
        ui.print_message(constants.Validation.CONFIG_ISSUES_FOUND, "warning")
        for issue in issues:
            ui.console.print(f"  [warning]⚠️  {issue}[/warning]")
    else:
        ui.print_message(constants.Validation.CONFIG_VALIDATION_PASSED, "success")

    ui.print_section(constants.Sections.INTEGRATION_CHECK, "swarm")
    status, details = system.check_wandb_status(constants)

    if "Not" in status:
        ui.print_message(f"wandb: {details}", "warning")
    else:
        ui.print_message(f"wandb: {details}", "success")

    ui.console.print()
    if issues or "Not" in status:
        ui.print_message(constants.Validation.VALIDATION_WITH_WARNINGS, "warning")
        ui.print_message(constants.Validation.REVIEW_ISSUES_TIP, "tip")
    else:
        ui.print_message(constants.Validation.ALL_VALIDATIONS_PASSED, "success")
        ui.print_message(constants.Validation.SYSTEM_READY, "success")


@cli.command("monitor")
def monitor(
    project: str = Option(
        constants.Wandb.DEFAULT_PROJECT,
        "--project", "-p",
        help="wandb project name to monitor"
    )
):
    """
    📈 Open wandb monitoring dashboard for the specified project.

    Quickly opens the wandb dashboard in your default browser to monitor
    training progress, view metrics, and analyze experiment results.
    """
    ui.print_header(
        constants.Headers.MONITOR_TITLE,
        constants.Headers.MONITOR_SUBTITLE_TPL.format(project=project)
    )

    url = system.get_wandb_url(constants, project)

    if not url:
        ui.print_message(
            constants.Messages.WANDB_UNAVAILABLE,
            "error"
        )
        raise Exit(1)

    ui.print_message(
        constants.Messages.BROWSER_LAUNCH_TPL.format(project=project),
        "swarm"
    )
    ui.print_wandb_info(project, url)
    ui.console.print()

    try:
        with ui.console.status(
            constants.Status.LAUNCHING_BROWSER,
            spinner="dots"
        ):
            sleep(0.5)
            open(url)
        ui.print_message(constants.Messages.BROWSER_SUCCESS, "success")
    except Exception as e:
        ui.print_message(constants.Messages.BROWSER_FAIL_TPL.format(e=e), "error")
        ui.print_message(constants.Messages.BROWSER_MANUAL_TPL.format(url=url), "info")


@cli.callback(invoke_without_command=True)
def main_callback(
    ctx: Context,
    version: bool | None = Option(
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
    if ctx.invoked_subcommand is None:
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