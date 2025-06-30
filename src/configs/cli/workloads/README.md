# CLI Configuration with Hydra

This directory contains the Hydra-based configuration for the Thermur CLI. The configuration system provides a clean separation between configuration data (schemas) and component construction (factories).

## Current Status

The CLI configuration has been fully migrated to the new schema-based system:

1. **Schemas** (`configs/cli/schemas/`): Define all configuration data using Pydantic models
2. **Factories** (`configs/cli/factories/`): Create Hydra-compatible builders for components  
3. **Workload** (`cli.py`): Composes all configurations using `make_config()`

## Migration Approach

The simplest migration path uses direct configuration access without complex adapters:

### Direct Integration Steps

1. **Update helper constructors** to accept DictConfig objects:
```python
class ThermurUI:
    def __init__(self, theme: DictConfig, ui: DictConfig):
        self.fire_gradient = theme.fire_gradient
        self.styles        = theme.styles
        self.panels        = ui.panels
```

2. **Use Hydra with Typer** via the zen decorator:
```python
@zen(cli_config).hydra_main(
    config_name  = "cli",
    version_base = None,
)
def main(cfg: DictConfig) -> None:
    app = typer.Typer(
        name = cfg.cli.app_name,
        help = cfg.cli.app_description,
    )
```

3. **Pass config through context** to commands:
```python
@app.command()
def info(ctx: typer.Context):
    cfg = ctx.obj
    ui = ThermurUI(cfg.theme, cfg.ui)
```

## Next Steps

When ready to complete the migration:

1. Choose migration approach (adapter vs. direct)
2. Update `cli.py` to use Hydra
3. Test all commands with new configuration
4. Remove `constants.py`
5. Update documentation

The configuration system is ready and waiting to be integrated!