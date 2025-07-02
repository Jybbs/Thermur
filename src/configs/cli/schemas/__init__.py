"""
Pydantic models defining the CLI configuration schema.

These models provide strict validation and type safety for all CLI configuration
options. Each model corresponds to a specific aspect of the CLI experience:
- Core models define the application structure and available commands
- UI models control visual presentation and theming
- System models handle environment detection and integration
- Message models manage user-facing text and prompts

All models use `extra="forbid"` to prevent configuration typos from going unnoticed.
"""
from .core     import *
from .messages import *
from .presets  import *
from .system   import *
from .ui       import *