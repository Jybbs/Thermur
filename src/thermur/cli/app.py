"""
Enhanced console-script target for Thermur, built with Typer and Rich.

This module provides the main CLI interface with visual feedback, system 
validation, configuration management, and seamless wandb integration.

The execution flow is:
1. User runs `thermur <command>` in the terminal
2. The `[project.scripts]` entry in `pyproject.toml` points to `src.__main__:cli`
3. `src.__main__.py` calls the `app` object from this file
4. Typer parses the command-line arguments and invokes the appropriate function
"""
import sys
import time
import typer
import webbrowser


from .ui   import *
from .constants import CLIConstants
from .explorer  import ConfigExplorer
from .prompts   import *
from .system    import *
from importlib  import metadata


app = typer.Typer(
    name                     = CLIConstants.Core.APP_NAME,
    help                     = CLIConstants.Core.APP_DESCRIPTION,
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
    
    print_header("Thermur", "Thermal Drone Swarm Training")
    
    # Get version info
    info = get_system_info()
    print_config_value("Version", f"v{info['thermur']}", "Thermur package version")
    print_config_value("Python",  f"v{info['python']}",  "Python runtime version")
    print_config_value("PyTorch", f"v{info['torch']}",   "Deep learning framework")
    console.print()
    
    # Display system info table
    table = create_system_table(console)
    console.print(table)
    
    # wandb status
    print_section("Integration Status", style="swarm")
    status, details = check_wandb_status()
    console.print(f"[swarm]📊 wandb: {status} • {details}[/swarm]")
    
    # Show quick start
    print_section("Quick Start", style="bright_green")
    
    for example in CLIConstants.Commands.EXAMPLES[:2]:
        print_command_example(example["desc"], example["command"], example["note"])
    
    raise typer.Exit()


@app.command("train")
def train(
    ctx: typer.Context,
    preset: str | None = typer.Option(
        None,
        "--preset", "-p",
        help="Configuration preset (quick, standard, large, debug)"
    ),
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive", "-i/-n",
        help="Enable interactive configuration prompts"
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Skip system checks and warnings"
    ),
    wandb_project: str | None = typer.Option(
        None,
        "--wandb-project", "-w",
        help="wandb project name for experiment tracking"
    ),
    config_overrides: list[str] | None = typer.Option(
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
        thermur train                                    # Interactive training
        thermur train --preset quick                     # Quick test run
        thermur train --config hyperparameters.lr=0.01  # Custom learning rate
        thermur train --wandb-project my-experiment     # Custom wandb project
        thermur train --no-interactive --force          # Non-interactive mode
    """
    print_header(
        "Thermur Training System",
        "Thermally-constrained drone swarm imitation learning"
    )
    
    # System validation
    if not force:
        _perform_system_validation()
    else:
        print_message("Skipping system checks (--force enabled)", "warning")
    
    # Configuration setup
    preset, wandb_project, config_overrides = _setup_configuration(
        preset           = preset,
        wandb_project    = wandb_project,
        config_overrides = config_overrides,
        interactive      = interactive,
        force            = force,
    )
    
    # Final confirmation
    if interactive:
        _confirm_training_setup(preset, wandb_project, config_overrides)
    
    # Start training
    _initiate_training(preset, config_overrides, wandb_project)


def _perform_system_validation():
    """
    Perform comprehensive system validation checks.
    
    Validates hardware capabilities, software versions, and integration status
    before proceeding with training initialization.
    """
    print_section("System Validation", "thermal")
    
    with console.status(
        "[thermal]Checking system requirements...[/thermal]", 
        spinner="dots"
    ):
        time.sleep(0.5)  # Brief pause for visual effect
        table = create_system_table(console)
    
    console.print(table)
    console.print()
    
    status, details = check_wandb_status()
    console.print(f"[swarm]📊 wandb: {status} • {details}[/swarm]")
    console.print()


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
    through interactive prompts or defaults.
    
    Args:
        preset           : Configuration preset name
        wandb_project    : wandb project name
        config_overrides : Hydra configuration overrides
        interactive      : Whether to use interactive prompts
        force            : Whether to skip validation
        
    Returns:
        Tuple of (preset, wandb_project, config_overrides)
    """
    print_section("Configuration Setup", "accent")
    
    # Select preset
    if interactive and not preset:
        preset = select_configuration_preset()
    
    if preset:
        print_message(
            f"Using preset: [bright_cyan]{preset}[/bright_cyan]", 
            "config"
        )
    
    # Setup wandb project
    if interactive and not wandb_project:
        wandb_project = ask_wandb_project_name()
    elif not wandb_project:
        wandb_project = CLIConstants.Wandb.DEFAULT_PROJECT
    
    # Handle configuration overrides
    if interactive and not config_overrides:
        additional = ask_for_config_overrides()
        config_overrides = (config_overrides or []) + additional
    
    # Validate configuration
    issues = validate_config_overrides(config_overrides)
    
    if issues and not force:
        _handle_configuration_issues(issues, interactive)
    
    return preset, wandb_project, config_overrides


def _handle_configuration_issues(issues: list[str], interactive: bool):
    """
    Handle configuration validation issues.
    
    Args:
        issues      : List of validation issues
        interactive : Whether to prompt for confirmation
    """
    if interactive and not confirm_system_override(issues):
        print_message("Training cancelled by user.", "warning")
        raise typer.Exit()
    elif not interactive:
        print_message("Configuration validation failed:", "error")
        for issue in issues:
            console.print(f"  • {issue}")
        print_message(
            "Use --force to override or fix the issues above.", 
            "info"
        )
        raise typer.Exit(1)


def _confirm_training_setup(
    preset           : str | None,
    wandb_project    : str,
    config_overrides : list[str] | None,
):
    """
    Show training summary and get final confirmation.
    
    Args:
        preset           : Configuration preset
        wandb_project    : wandb project name
        config_overrides : Configuration overrides
    """
    info = get_system_info()
    summary = {
        "preset"        : preset or "default",
        "wandb_project" : wandb_project,
        "overrides"     : len(config_overrides or []),
        "gpu_available" : info["cuda"],
    }
    
    if not show_training_summary(summary):
        print_message("Training cancelled by user.", "warning")
        raise typer.Exit()


def _initiate_training(
    preset           : str | None,
    config_overrides : list[str] | None,
    wandb_project    : str,
):
    """
    Initialize and start the training process.
    
    Args:
        preset           : Configuration preset
        config_overrides : Configuration overrides
        wandb_project    : wandb project name
    """
    print_section("Initializing Training", "thermal")
    
    url = get_wandb_url(wandb_project)
    print_wandb_info(wandb_project, url)
    console.print()
    
    try:
        _run_training(preset, config_overrides, wandb_project)
    except KeyboardInterrupt:
        console.print()
        print_message("Training interrupted by user.", "warning")
        raise typer.Exit()
    except Exception as e:
        print_message(f"Training failed: {str(e)}", "error")
        raise typer.Exit(1)


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
    print_message("Loading training components...", "info")
    
    # Build Hydra configuration
    hydra_train = _build_hydra_configuration(
        preset           = preset,
        config_overrides = config_overrides,
        wandb_project    = wandb_project,
    )
    
    console.print()
    
    # Execute with config
    if preset or config_overrides:
        args = []
        if preset:
            args.append(f"+preset={preset}")
        if config_overrides:
            args.extend(config_overrides)
        sys.argv.extend(args)
    
    hydra_train()


def _build_hydra_configuration(
    preset           : str | None,
    config_overrides : list[str] | None,
    wandb_project    : str,
):
    """
    Build Hydra configuration with lazy imports.
    
    Args:
        preset           : Configuration preset
        config_overrides : Configuration overrides
        wandb_project    : wandb project name
        
    Returns:
        Hydra-decorated training function
    """
    with create_progress() as progress:
        task = progress.add_task("Initializing core modules...", total=100)
        
        # Lazy imports
        progress.update(
            task, 
            advance     = 20, 
            description = "Loading configuration system..."
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
            description = "Registering configurations..."
        )
        register_configs()
        
        progress.update(
            task, 
            advance     = 50, 
            description = "Preparing Hydra runtime..."
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
        
        progress.update(task, advance=100, description="Ready to train!")
    
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
    
    Args:
        cfg                      : Resolved Hydra configuration
        instantiate              : Hydra-zen instantiate function
        pydantic_parser          : Pydantic parser for instantiation
        configure_loguru         : Loguru configuration function
        set_seed                 : Random seed setting function
        train_imitation_learning : Main training function
    """
    print_section("Building Training Components", "thermal")
    
    # Setup
    configure_loguru(
        instantiate(cfg.monitoring.logging, _parser=pydantic_parser)
    )
    set_seed(
        instantiate(cfg.hyperparameters, _parser=pydantic_parser).seed
    )
    
    # Build components
    components = _instantiate_training_components(
        cfg             = cfg,
        instantiate     = instantiate,
        pydantic_parser = pydantic_parser,
    )
    
    print_message("All components initialized successfully!", "success")
    console.print()
    
    # Launch training
    print_section("Training Started", "thermal")
    print_message("Monitoring thermal constraints and swarm dynamics", "thermal")
    print_message("Track progress in your wandb dashboard", "swarm")
    console.print()
    
    train_imitation_learning(**components)
    
    console.print()
    print_header("Training Complete! 🎉", "Your thermal swarm has learned to fly")


def _instantiate_training_components(cfg, instantiate, pydantic_parser):
    """
    Instantiate all training components with progress tracking.
    
    Args:
        cfg             : Resolved Hydra configuration
        instantiate     : Hydra-zen instantiate function
        pydantic_parser : Pydantic parser
        
    Returns:
        Dictionary of instantiated components
    """
    with create_progress() as progress:
        task = progress.add_task("Instantiating components...", total=9)
        
        components = {}
        component_configs = [
            ("environment",       "simulation",        "🌍 Environment"),
            ("expert_policy",     "expert_policy",     "🎓 Expert Policy"),
            ("policy",            "policy",            "🧠 Learning Policy"),
            ("data_collector",    "data_collector",    "📊 Data Collector"),
            ("experience_buffer", "experience_buffer", "💾 Experience Buffer"),
            ("loss_function",     "loss_function",     "📏 Loss Function"),
            ("optimizer",         "optimizer",         "⚙️  Optimizer"),
            ("hyperparameters",   "hyperparameters",   "🎛️  Hyperparameters"),
            ("wandb_config",      "monitoring.wandb",  "📊 wandb Tracking"),
        ]
        
        for i, (key, config_path, display_name) in enumerate(component_configs):
            progress.update(
                task, 
                completed   = i, 
                description = f"Setting up {display_name}..."
            )
            time.sleep(0.2)  # Brief pause for visual effect
            
            # Handle nested configs
            if "." in config_path:
                parent, child = config_path.split(".")
                config_obj = getattr(cfg, parent)[child]
            else:
                config_obj = getattr(cfg, config_path)
            
            components[key] = instantiate(config_obj, _parser=pydantic_parser)
        
        progress.update(task, completed=9)
        
        # Optional visualizer
        if hasattr(cfg, 'visualization'):
            components['visualizer'] = instantiate(
                cfg.visualization, 
                _parser=pydantic_parser
            )
        else:
            components['visualizer'] = None
    
    return components


@app.command("configure")
def configure():
    """
    🔧 Interactive configuration explorer and editor.
    
    Navigate the configuration hierarchy interactively, view Pydantic schemas,
    and edit values without manually writing Hydra overrides.
    """
    explorer  = ConfigExplorer()
    overrides = explorer.explore_interactive()
    
    if overrides:
        console.print()
        print_header("Generated Configuration Overrides")
        
        print_message("Add these to your training command:", "info")
        console.print()
        
        # Build command
        cmd_parts = ["thermur train"]
        for override in overrides:
            cmd_parts.append(f'--config "{override}"')
        
        command = " ".join(cmd_parts)
        console.print(f"[bold accent]$ {command}[/bold accent]")
        console.print()
        
        # Also show as separate lines
        print_message("Or use them individually:", "tip")
        for override in overrides:
            print_config_value("--config", override)
    else:
        print_message("No configuration changes made.", "info")


@app.command("info")
def info():
    """
    📊 Display comprehensive system and configuration information.
    
    Shows detailed information about the current system setup, including
    hardware capabilities, software versions, and configuration status.
    """
    print_header("Thermur System Information")
    
    # System diagnostics
    print_section("System Diagnostics", "thermal")
    table = create_system_table(console)
    console.print(table)
    
    # wandb status
    print_section("Integration Status", "swarm")
    status, details = check_wandb_status()
    print_wandb_info(CLIConstants.Wandb.DEFAULT_PROJECT, get_wandb_url())
    console.print(f"[muted]Status: {status} • {details}[/muted]")
    
    # Features
    print_section("Features & Capabilities", "accent")
    features_table = create_feature_table()
    console.print(features_table)
    
    # Configuration info
    print_section("Configuration System", "config")
    print_config_value(
        "Config Path", 
        "configs/", 
        "Hydra configuration directory"
    )
    
    preset_names = ", ".join(CLIConstants.Presets.CONFIGS.keys())
    print_config_value(
        "Presets", 
        preset_names, 
        "Available training presets"
    )
    
    print_config_value(
        "Explorer", 
        "thermur configure", 
        "Interactive configuration tool"
    )
    
    # Show tips
    print_training_tips()


@app.command("validate")
def validate(
    config_overrides: list[str] | None = typer.Option(
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
    print_header("System Validation", "Pre-flight checks for training")
    
    # System validation
    print_section("Hardware & Software", "thermal")
    
    with console.status(
        "[thermal]Running system diagnostics...[/thermal]", 
        spinner="dots"
    ):
        time.sleep(0.5)
        table = create_system_table(console)
    
    console.print(table)
    
    # Config validation
    print_section("Configuration Check", "config")
    
    with console.status(
        "[accent]Validating configuration...[/accent]", 
        spinner="dots"
    ):
        time.sleep(0.3)
        issues = validate_config_overrides(config_overrides)
    
    if issues:
        print_message("Configuration issues found:", "warning")
        for issue in issues:
            console.print(f"  [warning]⚠️  {issue}[/warning]")
    else:
        print_message("Configuration validation passed!", "success")
    
    # wandb validation
    print_section("Integration Check", "swarm")
    status, details = check_wandb_status()
    
    if "Not" in status:
        print_message(f"wandb: {details}", "warning")
    else:
        print_message(f"wandb: {details}", "success")
    
    # Summary
    console.print()
    if issues or "Not" in status:
        print_message("⚠️  Validation completed with warnings", "warning")
        print_message("Review the issues above before training", "tip")
    else:
        print_message("✅ All validations passed!", "success")
        print_message("Your system is ready for training", "success")


@app.command("monitor")
def monitor(
    project: str = typer.Option(
        CLIConstants.Wandb.DEFAULT_PROJECT,
        "--project", "-p",
        help="wandb project name to monitor"
    )
):
    """
    📈 Open wandb monitoring dashboard for the specified project.
    
    Quickly opens the wandb dashboard in your default browser to monitor
    training progress, view metrics, and analyze experiment results.
    """
    print_header("wandb Monitoring", f"Project: {project}")
    
    url = get_wandb_url(project)
    
    if not url:
        print_message(
            "wandb not available - install with 'pip install wandb'", 
            "error"
        )
        raise typer.Exit(1)
    
    print_message(
        f"Opening dashboard for project: [bright_cyan]{project}[/bright_cyan]", 
        "swarm"
    )
    print_wandb_info(project, url)
    console.print()
    
    try:
        with console.status(
            "[swarm]Launching browser...[/swarm]", 
            spinner="dots"
        ):
            time.sleep(0.5)
            webbrowser.open(url)
        print_message("Dashboard opened in your default browser!", "success")
    except Exception as e:
        print_message(f"Failed to open browser: {str(e)}", "error")
        print_message(f"Please visit manually: {url}", "info")


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: bool | None = typer.Option(
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
        # Show a nice welcome screen when no command is given
        print_header("Welcome to Thermur", "Thermal Drone Swarm Training")
        
        # Show available commands
        print_section("Available Commands", "accent")
        
        for cmd_info in CLIConstants.Commands.AVAILABLE:
            console.print(
                f"  {cmd_info['icon']} [bold accent]{cmd_info['name']:10}"
                f"[/bold accent] [muted]{cmd_info['desc']}[/muted]"
            )
        
        # Show examples
        print_section("Getting Started", "bright_green")
        
        for example in CLIConstants.Commands.EXAMPLES:
            print_command_example(
                example["desc"], 
                example["command"], 
                example["note"]
            )
        
        console.print()
        print_message("Ready to train some thermal swarms? 🔥", "thermal")
