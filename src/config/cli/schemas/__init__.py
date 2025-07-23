"""
Pydantic models defining the CLI configuration schema.

These models provide strict validation and type safety for all CLI configuration
options. Models are organized by functional cohesion:
- Application models define core configuration, commands, and integrations
- Display models control visual presentation, theming, and all messages
- Interaction models handle user prompts and preset configurations
- Secrets models manage secure OAuth2 token storage using BaseSettings

All models use `extra="forbid"` to prevent configuration typos from going unnoticed.
"""
