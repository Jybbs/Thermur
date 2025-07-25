# Thermur Style Guide

This document outlines the coding style and conventions used in the Thermur project. Maintaining a consistent style across the codebase enhances readability, simplifies maintenance, and reduces cognitive load when switching between files.

## Overarching Principles

- **Be concise**: Favor tidy, succinct logical structures without compromising legibility
- **Be clear**: Use self-documenting code with descriptive variable names and rich docstrings
- **Be consistent**: Follow established patterns in the codebase
- **Be modern**: Leverage language features and library utilities appropriately
- **Be mathematical**: Use proper mathematical notation in documentation
- **Maximum line length**: Keep lines under 88 characters

## Project Organization

### Module Structure

- Organize modules by functionality, with related code grouped together
- Keep imports at the usage site rather than re-exporting through `__init__.py`
- Avoid `__all__` declarations and `from x import *` patterns
- Place high-level functionality in the root of the package, with utilities in submodules

```python
# ✅ Good module organization
"""
Hydra-zen configuration factories for imitation learning components.

This package provides builders that create Hydra-compatible configurations
for instantiating components needed for imitation learning training.
"""
# Keep this file minimal - users import directly from submodules
```

### Package Responsibility

- Each package should have a clearly defined responsibility
- Module docstrings should explain the purpose and context of the module
- Related functionality should be grouped into coherent packages
- Utility functions should be separated into appropriate utility modules

```python
# ✅ Clear module responsibility
"""
Utilities for ensuring reproducible results via random seeding.
"""
from loguru import logger
from numpy  import random
from torch  import backends, cuda, manual_seed


def set_seed(seed: int):
    """
    Sets the random seed for all relevant libraries.

    This function seeds `random`, `numpy`, and `torch` to ensure that any
    stochastic processes in the application are repeatable.

    Args:
        seed: The integer seed to use.
    """
    # Implementation...
```

## Documentation Standards

### Module Docstrings

- Every module should begin with a docstring explaining its purpose
- Include context about how the module fits into the larger system
- Mention any key concepts, patterns, or algorithms implemented
- Keep module docstrings focused on the "what" and "why" rather than the "how"

```python
# ✅ Good module docstring
"""
Defines the Graph Neural Network (GNN) policy module, π_θ.

This file contains the implementation of the `torch.nn.Module` that serves as
the agent's brain. The policy, denoted π_θ, is a GNN designed to process the
flock's state, which is naturally represented as a dynamic graph. It learns to
output a nominal velocity command, 𝐮_nom, for each agent.

The architecture is explicitly designed to be configurable and to consume
`torch_geometric.data.Data` objects, which are generated from the environment's
`TensorDict` observations.
"""
```

### Class and Function Docstrings

- Rich, descriptive docstrings take precedence over inline comments
- Docstrings should fully describe behavior, parameters, return values, and exceptions
- Use Unicode mathematics notation (e.g., τ, Δt, θ) in docstrings to enhance clarity
- Include mathematical formulations with proper notation in docstrings
- Use inline comments only when necessary to explain non-obvious implementation details

```python
# ✅ Good function docstring with mathematical notation
def _compute_separation(self, position: Tensor) -> Tensor:
    """
    Calculates the separation force vector for each agent.
    
    The separation term implements the second of Reynolds' flocking rules,
    creating a repulsive force that prevents collisions between agents.
    For each agent i and its neighbor j, we calculate a repulsion vector:
    
        𝐅_sepᵢⱼ = (𝐱ᵢ - 𝐱ⱼ) / ||𝐱ᵢ - 𝐱ⱼ||²
        
    The total separation force is the sum of these repulsions:
    
        𝐅_sep = Σⱼ∈N(i) 𝐅_sepᵢⱼ
        
    The force magnitude is inversely proportional to the squared distance,
    creating a stronger repulsion between agents that are close to each other.
    
    Args:
        position : Tensor [N, dim] containing agent positions 𝐱
    
    Returns:
        Tensor [N, dim] of separation force vectors for all agents
    """
```

### Mathematical Notation

- Use Unicode symbols for mathematical variables and operators
- Format equations on separate lines with proper indentation for clarity
- Use subscripts and superscripts to match conventional notation (e.g., xᵢⱼ)
- Explain the meaning of mathematical symbols in the surrounding text
- Be consistent with notation across related functions and modules

```python
# ✅ Good mathematical notation in docstring
"""
This controller computes a desired velocity for each agent by summing forces
derived from the negative gradient of several potential functions, where the 
individual potential components follow classical Reynolds rules:
    - U_coh^(i)     = (1/2) · Σⱼ∈N(i) ||𝐱ᵢ - 𝐱ⱼ||²
    - U_sep^(i)     = Σⱼ∈N(i) 1/||𝐱ᵢ - 𝐱ⱼ||
    - U_align^(i,j) = (1/2) · ||𝐯ᵢ - 𝐯ⱼ||²
    - U_therm^(i)   = 1/(T_max - T_i)

The nominal control action is then 𝐮_nom^(i) = -∇ₓᵢU(𝐒ₜ)
"""
```

## Code Organization

### Import Structure

- Alphabetize all imports, with `from` imports typically appearing before standalone `import` statements due to alphabetical ordering
- Align import statements for readability with a single space between the longest import and its preceding `from` package
- Group related imports visually with blank lines if it improves clarity
- Avoid unnecessary imports; only import what you use

```python
# ✅ Correct structure and alignment
from collections                         import Counter, defaultdict
from config.imitation.schemas.monitoring import MonitoringModel
from pathlib                             import Path
from pytorch_lightning                   import LightningModule
from tensordict                          import TensorDict, TensorDictBase
from time                                import perf_counter
from torch                               import where
from torchrl.envs                        import EnvBase
from typing                              import Any, Callable, Optional

import mujoco
import torch
import wandb
```

### Variable Naming

- Use descriptive, self-documenting variable names
- Avoid single-letter variables except in mathematical contexts (e.g., x, y, z for coordinates)
- Create intermediate variables only when they improve readability, are used multiple times, 
  or clarify complex expressions
- Be consistent with variable naming across related functions

```python
# ✅ Clear, descriptive variable names and efficient structure
observation_dict    = self.observation_spec.zero()
formation           = self.config.flock.initial_formation
agent_count         = self.config.flock.agent_count
spatial_dims        = self.config.flock.spatial_dims
communication_range = self.config.flock.communication_range
formation_scale     = self.config.flock.formation_scale_factor

# ✅ Only create intermediate variables when they add clarity or are reused
rel_pos  = position[self._edge_source] - position[self._edge_target]
distance = torch.norm(input=rel_pos, dim=1, keepdim=True)
distance = torch.clamp(distance, min=self.flocking_params.min_distance)
```

### Alignment and Formatting

- Align variable declarations, dictionary entries, and function parameters when it improves readability
- Align equals signs, colons, and similar syntax elements when they appear in groups
- Use spaces around operators and after commas

```python
# ✅ Proper alignment in various contexts
# Variable declaration alignment
observation_dict    = self.observation_spec.zero()
formation           = self.config.flock.initial_formation
agent_count         = self.config.flock.agent_count

# Dictionary alignment
return {
    "model" : model, 
    "data"  : mujoco.MjData(model)
}

# Function call parameter alignment
observation_dict.set(
    key  = "position", 
    item = positions * communication_range * formation_scale
)

# Function declaration parameter alignment
def __init__(
    self,
    config,
    data_source        : Callable,
    compute_edge_index : Callable,
    observation_spec   : TensorDictBase,
    action_spec        : TensorDictBase,
    seed_fn            : Optional[Callable] = None,
):
```

### Whitespace and Line Breaks

- Use blank lines to separate logical sections of code
- Add newlines around conditional blocks for clarity
- Include a blank line after each method definition
- Avoid more than one consecutive blank line
- Add blank lines around if/else blocks

```python
# ✅ Proper whitespace usage
if formation == "cube":
    positions = self._generate_cube_formation(agent_count, spatial_dims)

else:
    positions = self._generate_sphere_formation(agent_count, spatial_dims)
```

## Configuration System

### Pydantic Models

- Use descriptive field names with detailed descriptions
- Align field parameters (default, gt, description) for readability
- Order fields logically based on importance or usage
- Use `extra="forbid"` to prevent unexpected fields
- Use validation constraints (gt, lt) where appropriate

```python
# ✅ Well-structured Pydantic model
class FlockModel(BaseModel, extra="forbid"):
    """
    Configures the collective properties and initial state of the agent flock.

    These parameters define the scale of the multi-agent system and the rules
    for local interaction. The `communication_range` is particularly critical
    as it defines the dynamic graph topology Gₜ = (V, Eₜ) at each timestep.
    This metric-based neighborhood is a practical starting point, while natural
    flocks often use a fixed topological neighborhood (e.g., 6-7 nearest agents).
    """
    agent_count: int = Field(
        default     = 30,
        gt          = 1,
        description = "The number of agents (N) in the flock."
    )
    communication_range: float = Field(
        default     = 50.0,
        gt          = 0,
        description = (
            "The metric distance in meters for defining the topological "
            "neighborhood graph."
        )
    )
    formation_scale_factor: float = Field(
        default     = 0.5,
        gt          = 0,
        description = (
            "Scaling factor applied to initial agent formations, as a fraction "
            "of the communication range. Controls the density of the flock."
        )
    )
```

### Hydra-zen Factory Patterns

- Use `builds` to create Hydra-compatible configurations
- Use the `populate_full_signature` parameter for full auto-completion
- Use `SI("${...}")` for configuration interpolation
- Follow the naming convention of `build_*` functions
- Maintain alignment in factory definitions

```python
# ✅ Well-structured Hydra-zen factory
build_environment = builds(
    SimulationEnv,
    action_spec             = build_action_spec,
    compute_edge_index      = compute_edge_index,
    config                  = build_composite_config,
    data_source             = build_data_source,
    observation_spec        = build_observation_spec,
    populate_full_signature = True,
    seed_fn                 = set_seed,
    zen_dataclass           = {
        "module"   : "src.configs.factories.environment",
        "cls_name" : "EnvironmentBuild"
    }
)
```

### Path Handling

- Use `pathlib.Path` for all path operations rather than string manipulation
- Use the `as_posix()` method when passing paths to libraries that require string paths
- Store paths in configuration where appropriate for flexibility

```python
# ✅ Correct path handling
model_path = self.config.environment.assets_dir / "flock.xml"
model = mujoco.MjModel.from_xml_path(model_path.as_posix())
```

## Implementation Patterns

### Type Hints

- Use type hints consistently in function signatures
- Align parameter type hints for readability
- Include return type hints
- Use `Optional` for parameters that may be None
- Use specific types over generic types when possible (e.g., `TensorDictBase` over `dict`)
- For signatures with a total of 2 parameters (like `self` and 1 parameter), keep them on a single line
- For signatures with 3+, place parameters on individual lines with aligned type hints
- Remember that `self` counts as a parameter when determining format
- Use built-in type annotations in Python 3.13 directly rather than importing from the typing module
    - e.g. Use `list`, `dict`; not `List`, `Dict`

```python
# ✅ Correct single-line signature (standalone function with 2 parameters)
def compute_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Calculate Euclidean distance between two points.
    """
    return torch.sqrt(torch.sum((x - y) ** 2))

# ✅ Correct single-line signature (method with self + 1 parameter)
def compute_gradient(self, temperature: torch.Tensor) -> torch.Tensor:
    """
    Compute temperature gradient at the given point.
    """
    return self._calculate_thermal_gradient(temperature)

# ✅ Correct multi-line signature (method with self + 2 parameters)
def initialize_agent(
    self,
    position : torch.Tensor,
    velocity : torch.Tensor, 
    temp     : float,
    state    : Optional[dict] = None
) -> Agent:
    """
    Initialize agent with the given parameters.
    """
    # Implementation follows...
```

```python
# ✅ Correct type hint usage
def _generate_sphere_formation(
    self, 
    n_agents : int, 
    dims     : int
) -> torch.Tensor:
    """
    Generates points distributed evenly on a sphere or circle.
    
    Args:
        n_agents : Number of agents to place
        dims     : Spatial dimensions (2 for circle, 3 for sphere)
        
    Returns:
        Tensor of shape [n_agents, dims] containing agent positions
    """
```

### Error Handling and Edge Cases

- Handle edge cases explicitly with clear documentation
- Use sensible defaults and fallbacks when appropriate
- Avoid silent failures
- Include informative error messages
- Use clamps and other safety mechanisms to prevent numerical issues

```python
# ✅ Proper handling of edge cases
if edge_index.numel() > 0:
    self._edge_source, self._edge_target = edge_index
else:
    # Handle empty graph case to prevent errors
    device            = edge_index.device
    self._edge_source = torch.tensor([], dtype=torch.long, device=device)
    self._edge_target = torch.tensor([], dtype=torch.long, device=device)

# ✅ Using clamps to prevent division by zero
distance = torch.clamp(distance, min=self.flocking_params.min_distance)
```

### Conditional Statements

- Prefer positive conditions over negative ones when possible
- Use clear, direct conditional expressions
- For simple cases with a clear fallback, consider using a default option
- Use ternary expressions for simple conditional assignments when it improves readability

```python
# ✅ Concise ternary expression (but keep under 88 characters)
gradient = grad_temp if grad_temp is not None else self._estimate_temperature_gradient(
    position=position, temperature=temperature
)

# ✅ Clean conditional with fallback
if formation == "cube":
    positions = self._generate_cube_formation(agent_count, spatial_dims)

else:  # Default to sphere formation
    positions = self._generate_sphere_formation(agent_count, spatial_dims)
```

### Method Structure

- Methods should be focused and single-purpose
- Organize methods by functionality and access level
- Private methods (with leading underscore) should appear after public methods
- Factory methods should use descriptive names that indicate what they create

```python
# General ordering within a class:
# 1. Class-level constants and variables
# 2. __init__ method
# 3. Public methods
# 4. Private methods (with _)
# 5. Static or class methods
```

### Dependency Injection

- Use constructor injection to pass dependencies
- Follow the existing pattern of providing dependencies rather than importing directly
- Use factories to construct complex objects with their dependencies
- Align parameters in instantiation calls

```python
# ✅ Constructor injection pattern
def __init__(
    self,
    agent_properties,
    flocking_params,
    reynolds_weights,
    safety_filter = None
):
    """
    Initializes the controller with the necessary configuration models.
    """
    self.agent_properties = agent_properties
    self.flocking_params  = flocking_params
    self.reynolds_weights = reynolds_weights
    self.safety_filter    = safety_filter

# ✅ Clean instantiation with aligned parameters
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
```

## Domain-Specific Patterns

### PyTorch and Tensor Operations

- Use torch operations that process entire tensors at once instead of loops
- Apply masks instead of conditionals inside loops
- Use PyTorch's broadcasting capabilities for element-wise operations
- Use specialized accumulation operations like `index_add_` over manual loops

```python
# ✅ Using masks for conditional operations
has_neighbors = (self._neighbor_count > 0).float().unsqueeze(1)
return (center_of_mass - position) * has_neighbors

# ✅ Using torch.where for conditional tensor assignment
grad_fallback = self._vertical_heat_gradient(position=position, temperature=temperature)
use_fallback  = (sig_counts == 0).unsqueeze(dim=1)
return torch.where(use_fallback, grad_fallback, grad_neighbors)

# ✅ Using specialized accumulation operations
center_of_mass = torch.zeros_like(position)
center_of_mass.index_add_(
    dim    = 0, 
    index  = self._edge_source, 
    source = position[self._edge_target]
)
```

### Tensor Dimension Management

- Be explicit about tensor dimensions in docstrings (e.g., `[N, dim]`)
- Use helper methods for dimension conversions when needed
- Be consistent with dimension handling across related operations
- Use descriptive variable names that indicate tensor shapes

```python
# ✅ Explicit dimension handling
def _ensure_1d_temperature(self, temperature: Tensor) -> Tensor:
    """
    Ensures temperature tensor is 1D by squeezing if it's [N, 1].
    
    Args:
        temperature: Tensor [N] or [N, 1] containing temperatures
            
    Returns:
        Tensor [N] with any singleton dimensions removed
    """
    if temperature.dim() > 1 and temperature.size(1) == 1:
        return temperature.squeeze(1)
    
    return temperature
```

### TensorDict Operations

- Use consistent parameter naming in `.get()` and `.set()` operations
- Align parameters for clarity
- Use explicit key/item naming for clarity

```python
# ✅ Clear, aligned TensorDict operations
observation_dict.set(
    key  = "position", 
    item = positions * communication_range * formation_scale
)

positions = observation_dict.get("position")
```

### Neural Network Architecture

- Organize networks into logical components (Encoder, Processor, Decoder)
- Use descriptive variable names for layers and activations
- Document the flow of data through the network
- Use dictionaries for configurable components like activation functions

```python
# ✅ Well-structured neural network
# Maps raw node features [𝐩, 𝐯, T, ∇T, E] to the hidden dimension.
self.encoder = Linear(in_dim, config.hidden_dim)

# A stack of GNN layers and recurrent cells for state updates.
self.convs = ModuleList()
self.grus  = ModuleList()
for _ in range(config.num_layers):
    self.convs.append(GCNConv(config.hidden_dim, config.hidden_dim))
    self.grus.append(GRUCell(config.hidden_dim, config.hidden_dim))

# Maps the final hidden state to a nominal action vector 𝐮_nom.
self.decoder = Linear(config.hidden_dim, out_dim)

# --- Activation Function ---
self.activation = {
    "relu" : ReLU, 
    "silu" : SiLU, 
    "tanh" : Tanh
}[config.activation]()
```

## CLI and Application Structure

### Command Organization

- Use Typer for CLI implementation
- Group related commands logically
- Use callback functions for flags that apply to all commands
- Implement lazy imports to keep CLI startup fast

```python
# ✅ Well-structured CLI command
@app.command()
def train():
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
        thermur train hyperparameters.learning_rate=0.001
        thermur train +experiment=large_flock
    """
    # Lazy imports to keep CLI startup fast
    from configs                        import register_configs, imitation_config
    from hydra_zen                      import instantiate, zen
    from hydra_zen.third_party.pydantic import pydantic_parser
    from thermur                        import (
        configure_loguru, 
        set_seed, 
        train_imitation_learning
    )
```

## XML and Configuration Files

- Organize XML elements logically
- Use consistent indentation (4 spaces)
- Align attributes for readability when they share context
- Use clear, descriptive comments
- Group related attributes together

```xml
<!-- ✅ Well-structured, aligned XML -->
<mujoco model="flock">
    <option 
        integrator = "RK4"
        timestep   = "0.05"
    >
        <flag 
            contact = "disable" 
            energy  = "enable" 
            gravity = "enable"
        />
    </option>
    
    <worldbody>
        <geom 
            name  = "ground" 
            pos   = "0 0 -0.1" 
            rgba  = "0.7 0.7 0.7 1"
            size  = "100 100 0.1" 
            type  = "plane"
        />
        
        <body 
            name = "drone_template" 
            pos  = "0 0 0"
        >
            <geom 
                mass = "0.5"
                name = "drone_geom" 
                rgba = "0.2 0.2 0.8 0.9" 
                size = "0.1" 
                type = "sphere"
            />
        </body>
    </worldbody>
</mujoco>
```

## Functional Programming Patterns

### Pure Functions

- Design functions to be free of side effects where possible
- Clearly document any state changes
- Return values rather than modifying arguments in-place when appropriate
- Use named parameters for clarity

```python
# ✅ Pure function design
def _generate_sphere_formation(
    self, 
    n_agents : int, 
    dims     : int
) -> torch.Tensor:
    """
    Generates points distributed evenly on a sphere or circle.
    
    For 3D, uses the Fibonacci sphere algorithm to generate points that are
    approximately equidistant on a sphere. For 2D, places points evenly on
    a circle using angular spacing.
    
    Args:
        n_agents : Number of agents to place
        dims     : Spatial dimensions (2 for circle, 3 for sphere)
            
    Returns:
        Tensor of shape [n_agents, dims] containing agent positions
    """
    if dims == 2:
        theta = torch.linspace(0, 2 * torch.pi, n_agents + 1)[:-1]
        x     = torch.cos(theta)
        y     = torch.sin(theta)
        
        return torch.stack([x, y], dim=1)
    else:
        phi     = (1 + 5 ** 0.5) / 2  # Golden ratio
        indices = torch.arange(0, n_agents, dtype=torch.float32)
        theta   = 2 * torch.pi * indices / phi
        z       = 1 - (2 * indices + 1) / n_agents
        radius  = torch.sqrt(1 - z * z)
        x       = radius * torch.cos(theta)
        y       = radius * torch.sin(theta)
        
        return torch.stack([x, y, z], dim=1)
```

### Helper Methods

- Extract common operations into private helper methods
- Keep methods focused on a single responsibility
- Use meaningful names that describe what the method does
- Group related functionality together

```python
# ✅ Focused helper method with clear purpose
def _ensure_1d_temperature(self, temperature: Tensor) -> Tensor:
    """
    Ensures temperature tensor is 1D by squeezing if it's [N, 1].
    
    Args:
        temperature: Tensor [N] or [N, 1] containing temperatures
            
    Returns:
        Tensor [N] with any singleton dimensions removed
    """
    if temperature.dim() > 1 and temperature.size(1) == 1:
        return temperature.squeeze(1)
    
    return temperature
```

### Shared State Management

- Initialize shared state in a centralized location
- Reset state when necessary to prevent errors
- Document the purpose and lifecycle of shared state

```python
# ✅ Clear state management pattern
def _reset_shared_state(self):
    """
    Resets the shared graph state variables to None.
    """
    self._edge_source    = None
    self._edge_target    = None
    self._neighbor_count = None
    self._safe_count     = None

def _update_graph_state(self, edge_index: Tensor, num_agents: int):
    """
    Updates shared state for graph calculations across Reynolds rules.
    
    Args:
        edge_index : Tensor defining the communication graph topology Gₜ = (V, Eₜ)
        num_agents : The total number of agents N in the flock
    """
    # Implementation follows...
```

## Commit Message Conventions

Follow the Conventional Commits specification for commit messages:

```
type(scope): subject

body

footer
```

- **type**: feat, fix, docs, style, refactor, test, chore
- **scope**: optional, indicates section of codebase (e.g., simulation, models)
- **subject**: concise description in present tense
- **body**: optional, detailed description with motivation and changes
- **footer**: optional, references issues or breaking changes

Example:
```
feat(simulation): implement physics initialization and agent formation logic

- Create MuJoCo XML model for drone flock with dynamic physics configuration
- Add assets_dir field to EnvironmentModel for configurable simulation assets path
- Add formation_scale_factor to FlockModel to control initial flock density

Resolves: #5 (partially - step logic still to be implemented)
