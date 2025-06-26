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
import webbrowser

from importlib import metadata

import typer

from .console import (
    console,
    create_feature_table,
    create_progress,
    print_command_example,
    print_config_value,
    print_header,
    print_message,
    print_section,
    print_training_tips,
    print_wandb_info,
)
from .explorer import ConfigExplorer
from .prompts  import (
    ask_for_config_overrides,
    ask_wandb_project_name,
    confirm_system_override,
    select_configuration_preset,
    show_training_summary,
)
from .system   import (
    check_wandb_status,
    create_system_table,
    get_system_info,
    get_wandb_url,
    validate_config_overrides,
)


app = typer.Typer(
    name             = "thermur",
    help             = "🔥 Thermally-constrained drone swarm training toolkit",
    add_completion   = False,
    rich_markup_mode = "rich",
    no_args_is_help  = True,
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
    print_command_example(
        "Start interactive training",
        "thermur train",
        "Guides you through configuration"
    )
    print_command_example(
        "Run with a preset",
        "thermur train --preset quick",
        "Perfect for testing"
    )
    
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
    
    This command provides a comprehensive training workflow with system validation,
    configuration management, and seamless wandb integration.
    
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
    print_section("System Validation", "thermal")
    
    if not force:
        with console.status("[thermal]Checking system requirements...[/thermal]", spinner="dots"):
            time.sleep(0.5)  # Brief pause for visual effect
            table = create_system_table(console)
        
        console.print(table)
        console.print()
        
        status, details = check_wandb_status()
        console.print(f"[swarm]📊 wandb: {status} • {details}[/swarm]")
        console.print()
    else:
        print_message("Skipping system checks (--force enabled)", "warning")
    
    # Configuration setup
    print_section("Configuration Setup", "accent")
    
    if interactive and not preset:
        preset = select_configuration_preset()
    
    if preset:
        print_message(f"Using preset: [bright_cyan]{preset}[/bright_cyan]", "config")
    
    if interactive and not wandb_project:
        wandb_project = ask_wandb_project_name()
    elif not wandb_project:
        wandb_project = "thermur"
    
    if interactive and not config_overrides:
        additional = ask_for_config_overrides()
        config_overrides = (config_overrides or []) + additional
    
    # Validate
    issues = validate_config_overrides(config_overrides)
    
    if issues and not force:
        if interactive and not confirm_system_override(issues):
            print_message("Training cancelled by user.", "warning")
            raise typer.Exit()
        elif not interactive:
            print_message("Configuration validation failed:", "error")
            for issue in issues:
                console.print(f"  • {issue}")
            print_message("Use --force to override or fix the issues above.", "info")
            raise typer.Exit(1)
    
    # Final confirmation
    if interactive:
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
    
    # Start training
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
    
    with create_progress() as progress:
        task = progress.add_task("Initializing core modules...", total=100)
        
        # Lazy imports
        progress.update(task, advance=20, description="Loading configuration system...")
        from configs import imitation_config, register_configs
        from hydra_zen import instantiate, zen
        from hydra_zen.third_party.pydantic import pydantic_parser
        from thermur import configure_loguru, set_seed, train_imitation_learning
        
        progress.update(task, advance=30, description="Registering configurations...")
        register_configs()
        
        # Build config args
        args = []
        if preset:
            args.append(f"+preset={preset}")
        if config_overrides:
            args.extend(config_overrides)
        
        progress.update(task, advance=50, description="Preparing Hydra runtime...")
        
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
            print_section("Building Training Components", "thermal")
            
            # Setup
            configure_loguru(instantiate(cfg.monitoring.logging, _parser=pydantic_parser))
            set_seed(instantiate(cfg.hyperparameters, _parser=pydantic_parser).seed)
            
            # Build components with progress
            with create_progress() as component_progress:
                component_task = component_progress.add_task("Instantiating components...", total=9)
                
                components = {}
                steps = [
                    ("simulation",        "🌍 Environment"),
                    ("expert_policy",     "🎓 Expert Policy"),
                    ("policy",            "🧠 Learning Policy"),
                    ("data_collector",    "📊 Data Collector"),
                    ("experience_buffer", "💾 Experience Buffer"),
                    ("loss_function",     "📏 Loss Function"),
                    ("optimizer",         "⚙️  Optimizer"),
                    ("hyperparameters",   "🎛️  Hyperparameters"),
                    ("monitoring.wandb",  "📊 wandb Tracking"),
                ]
                
                for i, (key, display_name) in enumerate(steps):
                    component_progress.update(
                        component_task, 
                        completed=i, 
                        description=f"Setting up {display_name}..."
                    )
                    time.sleep(0.2)  # Brief pause for visual effect
                    
                    if "." in key:  # Handle nested configs
                        parent, child = key.split(".")
                        components[key.replace(".", "_")] = instantiate(
                            getattr(cfg, parent)[child], 
                            _parser=pydantic_parser
                        )
                    else:
                        components[key] = instantiate(
                            getattr(cfg, key), 
                            _parser=pydantic_parser
                        )
                
                component_progress.update(component_task, completed=9)
                
                # Optional visualizer
                if hasattr(cfg, 'visualization'):
                    components['visualizer'] = instantiate(cfg.visualization, _parser=pydantic_parser)
                else:
                    components['visualizer'] = None
            
            print_message("All components initialized successfully!", "success")
            console.print()
            
            # Launch training
            print_section("Training Started", "thermal")
            print_message("Monitoring thermal constraints and swarm dynamics", "thermal")
            print_message("Track progress in your wandb dashboard", "swarm")
            console.print()
            
            train_imitation_learning(
                environment       = components['simulation'],
                expert_policy     = components['expert_policy'],
                policy            = components['policy'],
                data_collector    = components['data_collector'],
                experience_buffer = components['experience_buffer'],
                loss_function     = components['loss_function'],
                optimizer         = components['optimizer'],
                hyperparameters   = components['hyperparameters'],
                wandb_config      = components['monitoring_wandb'],
                visualizer        = components['visualizer'],
            )
            
            console.print()
            print_header("Training Complete! 🎉", "Your thermal swarm has learned to fly")
        
        progress.update(task, advance=100, description="Ready to train!")
    
    console.print()
    
    # Execute with config
    if args:
        sys.argv.extend(args)
    
    hydra_train()


@app.command("configure")
def configure():
    """
    🔧 Interactive configuration explorer and editor.
    
    Navigate the configuration hierarchy interactively, view Pydantic schemas,
    and edit values without manually writing Hydra overrides. This command
    provides a user-friendly way to explore and customize training configurations.
    
    The explorer will:
    - Dynamically discover all available configurations
    - Display schema fields in rich tables with descriptions
    - Allow interactive editing of configuration values
    - Generate proper Hydra overrides for the changes
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
    print_wandb_info("thermur", get_wandb_url())
    console.print(f"[muted]Status: {status} • {details}[/muted]")
    
    # Features
    print_section("Features & Capabilities", "accent")
    features_table = create_feature_table()
    console.print(features_table)
    
    # Configuration info
    print_section("Configuration System", "config")
    print_config_value("Config Path", "configs/", "Hydra configuration directory")
    print_config_value("Presets", "quick, standard, large, debug", "Available training presets")
    print_config_value("Explorer", "thermur configure", "Interactive configuration tool")
    
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
    
    with console.status("[thermal]Running system diagnostics...[/thermal]", spinner="dots"):
        time.sleep(0.5)
        table = create_system_table(console)
    
    console.print(table)
    
    # Config validation
    print_section("Configuration Check", "config")
    
    with console.status("[accent]Validating configuration...[/accent]", spinner="dots"):
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
        "thermur",
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
        print_message("wandb not available - install with 'pip install wandb'", "error")
        raise typer.Exit(1)
    
    print_message(f"Opening dashboard for project: [bright_cyan]{project}[/bright_cyan]", "swarm")
    print_wandb_info(project, url)
    console.print()
    
    try:
        with console.status("[swarm]Launching browser...[/swarm]", spinner="dots"):
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
        
        commands = [
            ("train",     "🔥", "Train a thermal drone swarm with imitation learning"),
            ("configure", "🔧", "Interactively explore and edit configurations"),
            ("info",      "📊", "Display system information and capabilities"),
            ("validate",  "✅", "Validate system setup and configuration"),
            ("monitor",   "📈", "Open wandb dashboard to monitor experiments"),
        ]
        
        for cmd, icon, desc in commands:
            console.print(f"  {icon} [bold accent]{cmd:10}[/bold accent] [muted]{desc}[/muted]")
        
        # Show examples
        print_section("Getting Started", "bright_green")
        
        print_command_example(
            "Start your first training run",
            "thermur train --preset quick",
            "Perfect for testing the system"
        )
        
        print_command_example(
            "Check your system setup",
            "thermur info",
            "See what's installed and ready"
        )
        
        print_command_example(
            "Get help on any command",
            "thermur train --help",
            "Detailed usage information"
        )
        
        console.print()
        print_message("Ready to train some thermal swarms? 🔥", "thermal")
