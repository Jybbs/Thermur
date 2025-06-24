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
from importlib import metadata
from rich      import print
from typer     import Context, Exit, Option, Typer
from typing    import Optional

app = Typer(
    name           = "thermur",
    help           = "A toolkit for simulating thermally-constrained drone swarms.",
    add_completion = False,
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
        print(f"[bold green]Thermur[/] version: {metadata.version("thermur")}")
        raise Exit()


@app.command()
def train(
    visualize: bool = Option(
        False,
        "--visualize",
        help="Enable real-time 3D visualization of the simulation."
    )
):
    """
    Train the GNN policy using imitation learning.

    This command initializes and runs the main training loop. It uses Hydra for
    configuration management, allowing for easy overrides of any parameter via
    the command line.

    The necessary libraries for training (Hydra, PyTorch, etc.) are imported
    within this function ('lazily') to keep the main CLI startup time fast
    for simple commands like `--version`.

    Example:
        thermur train
        thermur train --visualize
        thermur train hyperparameters.learning_rate=0.001
        thermur train +experiment=large_swarm
    """
    from configs                        import register_configs, imitation_config
    from hydra_zen                      import instantiate, zen
    from hydra_zen.third_party.pydantic import pydantic_parser
    from thermur                        import (
        configure_loguru, 
        set_seed, 
        train_imitation_learning
    )

    register_configs()

    @zen(imitation_config).hydra_main(
        config_name  = "train",
        config_path  = None,
        version_base = None,
        with_log_configuration = False,
    )
    def hydra_train(cfg):
        # Apply CLI visualization flag to override config if specified
        if visualize:
            cfg.visualization.enabled = True
        """
        The core training function, wrapped by Hydra.

        This function is the main entry point for the training process after
        Hydra has parsed and composed the configuration. It receives the final
        configuration object (`cfg`) and proceeds to instantiate all necessary
        components—from the environment and policies to the optimizer and data
        collectors—before launching the training loop.

        Args:
            cfg: The fully-resolved Hydra configuration object, built from
                 the structures defined in the `configs` modules.
        """
        print("[bold green]Starting Thermur imitation learning training[/]")
        
        configure_loguru(instantiate(cfg.logging, _parser=pydantic_parser))
        set_seed(instantiate(cfg.hyperparameters, _parser=pydantic_parser).seed)

        print("[yellow]Instantiating components...[/]")
        components = {
            "environment"       : instantiate(cfg.environment,       _parser=pydantic_parser),
            "expert_policy"     : instantiate(cfg.expert_policy,     _parser=pydantic_parser),
            "policy"            : instantiate(cfg.policy,            _parser=pydantic_parser),
            "data_collector"    : instantiate(cfg.data_collector,    _parser=pydantic_parser),
            "experience_buffer" : instantiate(cfg.experience_buffer, _parser=pydantic_parser),
            "loss_function"     : instantiate(cfg.loss_function,     _parser=pydantic_parser),
            "optimizer"         : instantiate(cfg.optimizer,         _parser=pydantic_parser),
            "hyperparameters"   : instantiate(cfg.hyperparameters,   _parser=pydantic_parser),
            "wandb_config"      : instantiate(cfg.wandb,             _parser=pydantic_parser),
        }
        
        # Initialize visualizer if enabled in config
        if cfg.visualization.enabled:
            print("[yellow]Initializing visualization...[/]")
            components["visualizer"] = instantiate(
                cfg.visualizer, 
                _parser=pydantic_parser,
            )
        
        # Launch the main imitation learning training loop with all components.
        print("[green]Starting training loop...[/]")
        train_imitation_learning(**components)

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
