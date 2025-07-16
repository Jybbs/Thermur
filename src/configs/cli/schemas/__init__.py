"""
Pydantic models defining the CLI configuration schema.

These models provide strict validation and type safety for all CLI configuration
options. Models are organized by functional cohesion:
- Application models define core configuration, commands, and integrations
- Display models control visual presentation, theming, and all messages
- Interaction models handle user prompts and preset configurations
- Token models manage OAuth2 token storage and validation

All models use `extra="forbid"` to prevent configuration typos from going unnoticed.
"""
from .application import *
from .display     import *
from .interaction import *
from .tokens      import *