"""
Configuration explorer schemas for the Thermur CLI.

This module defines models for the interactive configuration exploration
tool that allows users to navigate and modify Hydra configurations.
"""
from pydantic import BaseModel, Field, PositiveInt


class ExplorerModel(BaseModel, extra="forbid"):
    """
    Configures the interactive configuration explorer.
    
    This model defines the behavior of the configuration exploration tool,
    which allows users to navigate and modify Hydra configurations interactively.
    It includes display options, navigation settings, and field rendering rules.
    """
    enable_search: bool = Field(
        default     = True,
        description = "Whether to enable search functionality in explorer"
    )
    expand_depth: PositiveInt = Field(
        default     = 2,
        description = "Initial expansion depth for nested configurations"
    )
    indent_size: PositiveInt = Field(
        default     = 2,
        description = "Number of spaces for each indentation level"
    )
    max_description_length: PositiveInt = Field(
        default     = 80,
        description = "Maximum characters to display for field descriptions"
    )
    show_defaults: bool = Field(
        default     = True,
        description = "Whether to display default values in the explorer"
    )
    show_types: bool = Field(
        default     = True,
        description = "Whether to display field types in the explorer"
    )
    type_colors: dict[str, str] = Field(
        default = {
            "bool"  : "magenta",
            "dict"  : "blue",
            "float" : "cyan", 
            "int"   : "cyan",
            "list"  : "green",
            "str"   : "yellow"
        },
        description = "Color mapping for different field types"
    )


class ExplorerMessagesModel(BaseModel, extra="forbid"):
    """
    Messages specific to the configuration explorer functionality.
    
    This model contains all text displayed during interactive configuration
    exploration, including prompts, errors, and status messages.
    """
    cannot_explore: str = Field(
        default     = "Cannot explore target: {type_name}",
        description = "Error when target cannot be explored"
    )
    config_import_failed: str = Field(
        default     = "Failed to import configs module",
        description = "Error when configs module import fails"
    )
    dataclass_table_title: str = Field(
        default     = "Dataclass Fields",
        description = "Title for dataclass fields table"
    )
    default_workload_doc: str = Field(
        default     = "Workload configuration",
        description = "Default documentation for workloads"
    )
    edit_header_prefix: str = Field(
        default     = "Editing",
        description = "Prefix for edit mode headers"
    )
    explore_header_prefix: str = Field(
        default     = "Exploring",
        description = "Prefix for exploration mode headers"
    )
    field_not_found: str = Field(
        default     = "Field '{field_name}' not found",
        description = "Error when field is not found"
    )
    field_prompt: str = Field(
        default     = "[bold]Field to edit: [/bold]",
        description = "Prompt for field selection"
    )
    generic_component_name: str = Field(
        default     = "Configuration Component",
        description = "Generic name for config components"
    )
    header_title: str = Field(
        default     = "Interactive Configuration Explorer",
        description = "Main header for explorer"
    )
    max_depth_reached: str = Field(
        default     = "Reached maximum exploration depth",
        description = "Message when max depth is reached"
    )
    nested_component_name: str = Field(
        default     = "Nested Component",
        description = "Name for nested components"
    )
    no_workloads_found: str = Field(
        default     = "No workload configurations found",
        description = "Error when no workloads exist"
    )
    override_added: str = Field(
        default     = "Added override: {override}",
        description = "Confirmation when override is added"
    )
    overrides_generated: str = Field(
        default     = "Generated {count} configuration overrides",
        description = "Summary of generated overrides"
    )
    prompt_to_edit: str = Field(
        default     = "Enter field names to edit, or press Enter to finish.",
        description = "Instructions for editing fields"
    )
    schema_table_title: str = Field(
        default     = "Schema Fields",
        description = "Title for schema fields table"
    )
    unknown_config_type: str = Field(
        default     = "Unknown configuration type: {type_name}",
        description = "Error for unknown config types"
    )
    workload_component_name: str = Field(
        default     = "Workload",
        description = "Name for workload components"
    )