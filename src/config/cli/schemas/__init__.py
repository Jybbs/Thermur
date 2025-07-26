"""
Pydantic models defining the CLI configuration schema.

These models provide strict validation and type safety for all CLI configuration
options. Models are now organized to match the helper modules 1:1:
- globus: Download configuration and OAuth2 token storage
- prompts: Interactive prompts and training presets  
- system: System requirements and validation rules
- ui: Display settings, messages, and application metadata

All models use `extra="forbid"` to prevent configuration typos from going unnoticed.
"""
from .globus  import *
from .prompts import *
from .system  import *
from .ui      import *