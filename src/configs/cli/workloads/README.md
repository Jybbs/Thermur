# CLI Configuration Workload

This workload provides the complete configuration for the Thermur command-line interface using Hydra-zen and Pydantic schemas.

## Overview

The `cli.py` workload composes all CLI-related configurations into a single Hydra config that can be loaded and instantiated by the CLI application. It uses structured configuration interpolation to ensure consistency across all components.

## Usage

The CLI configuration is loaded automatically when running any Thermur command:

```bash
thermur train      # Loads CLI config for training interface
thermur info       # Loads CLI config for system information display
thermur validate   # Loads CLI config for validation checks
```

## Configuration Structure

The workload aggregates configurations from multiple schemas:

- **`cli`**: Core CLI settings (app name, description, flags)
- **`commands`**: Available commands and their metadata
- **`headers`**: Section titles and subtitles for different commands
- **`messages`**: All user-facing strings and templates
- **`message_types`**: Message styling with icons and colors
- **`prompts`**: Interactive prompt configurations
- **`sections`**: Section titles for content organization
- **`status`**: Status messages for progress indicators
- **`theme`**: Terminal styling with fire gradient colors
- **`training_components`**: Component mappings for training initialization
- **`ui`**: UI constants for Rich components
- **`wandb_display`**: Weights & Biases integration settings

## Customization

To override any CLI configuration value, you can:

1. **Create a custom config file**:
   ```yaml
   # config/custom_cli.yaml
   defaults:
     - cli
   
   theme:
     fire_gradient:
       - "#0000FF"  # Blue instead of red
       - "#00FFFF"  # Cyan gradient
   ```

2. **Use command-line overrides**:
   ```bash
   thermur train cli.debug_mode=true
   thermur info theme.styles.thermal="bold blue"
   ```

3. **Extend the workload**:
   ```python
   # In your own workload file
   from hydra_zen import make_config
   from .cli import cli_config
   
   custom_cli_config = make_config(
       defaults=["cli"],
       cli=dict(
           app_description="My Custom Thermur CLI"
       )
   )
   ```

## Integration with CLI Commands

The CLI loads this configuration once at startup and passes it through the Typer context:

```python
# In cli.py
cfg         = load_cli_config()  # Loads this workload
app_context = AppContext(cfg)
ctx.obj     = app_context

# In commands
def train(ctx: Context):
    cfg = ctx.obj.config
    ui  = ctx.obj.ui  # ThermurUI created with cfg.theme and cfg.ui
```

This design ensures consistent styling and messaging across all CLI commands while maintaining separation between configuration and implementation.