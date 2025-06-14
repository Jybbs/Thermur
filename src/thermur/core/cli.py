"""
Console-script target wired in pyproject.toml, built with Typer.

The main entry point is the `app` object. Commands are registered to it using
the `@app.command()` decorator.

The execution flow is as follows:
1. User runs `thermur` or `python -m src` in the terminal.
2. The `[project.scripts]` entry in `pyproject.toml` points to `src.__main__:cli`.
3. `src.__main__.py` calls the `app` object from this file.
4. `typer` parses the command-line arguments (e.g., `train`, `--version`).
5. `typer` invokes the appropriate function (e.g., `train()` or `version_callback`).
"""
from rich    import print
from typer   import Context, Exit, Option, Typer
from typing  import Optional

# Import version from parent package
from .. import __version__


# The docstring of this object will be used as the main `--help` text
app = Typer(
    name           = "thermur",
    help           = "A toolkit for simulating thermally-constrained drone swarms.",
    add_completion = False
)

def version_callback(value: bool):
    """
    A callback function triggered by the `--version` flag.

    `typer` invokes this function if '--version' or '-v' is present on the
    command line. If it is, we print the version and then call `typer.Exit()`
    to cleanly terminate the program before any other logic runs.

    Args:
        value: The boolean value of the flag (True if present).
    """
    if value:
        print(f"[bold green]Thermur[/] version: {__version__}")
        raise Exit()

@app.command()
def train():
    """
    Train the GNN policy using imitation learning.

    This command initializes the training configuration and runs the imitation
    learning training loop. Configuration is managed by Hydra, allowing for
    easy overrides via command line arguments.
    
    Example:
        thermur train
        thermur train hyperparameters.learning_rate=0.001
        thermur train +experiment=large_swarm
    """
    # Lazy imports to keep CLI fast
    from configs import register_configs
    from configs.train import train_config
    from hydra_zen import zen
    
    # Register all configurations with Hydra
    register_configs()
    
    # Use hydra-zen to run the training with proper config management
    @zen(train_config).hydra_main(
        config_name  = "train",
        config_path  = None,
        version_base = None,
    )
    def hydra_train(cfg):
        """Inner function that receives the Hydra config."""
        from hydra_zen import instantiate
        from hydra_zen.third_party.pydantic import pydantic_parser
        from thermur import configure_loguru, set_seed, train_imitation_learning
        
        print("[bold green]Starting Thermur imitation learning training[/]")
        
        # Setup logging and seed
        configure_loguru(instantiate(cfg.logging, _parser=pydantic_parser))
        set_seed(instantiate(cfg.hyperparameters, _parser=pydantic_parser).seed)
        
        # Instantiate all components
        print("[yellow]Instantiating components...[/]")
        components = {
            'environment': instantiate(cfg.environment, _parser=pydantic_parser),
            'expert_policy': instantiate(cfg.expert_policy, _parser=pydantic_parser),
            'policy': instantiate(cfg.policy, _parser=pydantic_parser),
            'data_collector': instantiate(cfg.data_collector, _parser=pydantic_parser),
            'experience_buffer': instantiate(cfg.experience_buffer, _parser=pydantic_parser),
            'loss_function': instantiate(cfg.loss_function, _parser=pydantic_parser),
            'optimizer': instantiate(cfg.optimizer, _parser=pydantic_parser),
            'hyperparameters': instantiate(cfg.hyperparameters, _parser=pydantic_parser),
            'wandb_config': instantiate(cfg.wandb, _parser=pydantic_parser),
        }
        
        # Run training
        print("[green]Starting training loop...[/]")
        train_imitation_learning(**components)
    
    # Execute the hydra-wrapped training
    hydra_train()

@app.callback()
def main_callback(
    ctx     : Context,
    version : Optional[bool] = Option(
        None,
        "--version",
        "-v",
        callback = version_callback,
        is_eager = True,
        help     = "Show the application's version and exit.",
    ),
):
    """
    The main application callback, run before any command.

    This function is automatically invoked by Typer before any specific command
    (like `train`) is run. It's the ideal place for global flags like
    `--version` that should apply to the application as a whole.
    """
    pass


def cli_main():
    """
    Main CLI entry point function.
    
    This function is called from __main__.py and exported for convenience.
    """
    app()
