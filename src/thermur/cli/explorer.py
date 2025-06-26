"""
Interactive configuration explorer for the Thermur CLI.

This module provides functionality to interactively browse and edit 
configuration schemas, allowing users to navigate the configuration
hierarchy and modify values without manually writing Hydra overrides.
"""
import inspect
import pkgutil

from importlib  import import_module
from pathlib    import Path

from hydra_zen  import builds, get_target, zen
from pydantic   import BaseModel
from rich.table import Table

from .console import console, print_header, print_message, print_section
from .prompts import prompt_for_field_value, select_config_component


class ConfigExplorer:
    """
    Interactive configuration explorer and editor.
    
    Dynamically discovers and navigates the configuration hierarchy,
    displays schema information in rich tables, and generates Hydra 
    overrides from user edits.
    """
    
    def __init__(self):
        """
        Initialize the configuration explorer.
        """
        self.overrides = []
    
    def explore_interactive(self) -> list[str]:
        """
        Launch interactive configuration exploration and editing.
        
        Dynamically discovers available configurations and guides the user
        through exploration and editing.
        
        Returns:
            List of Hydra configuration overrides generated from edits
        """
        print_header("Interactive Configuration Explorer")
        
        # Import and register configs
        try:
            from configs import register_configs
            register_configs()
        except ImportError:
            print_message("Failed to import configs module", "error")
            return []
        
        # Discover and select workload
        workload_config = self._select_workload()
        if not workload_config:
            return []
        
        # Explore the workload configuration tree
        self._explore_config_tree(workload_config, prefix="")
        
        return self.overrides
    
    def _select_workload(self) -> dict | None:
        """
        Dynamically discover and select a workload configuration.
        
        Returns:
            Selected workload configuration or None if cancelled
        """
        # Discover workloads from the workloads module
        workloads = self._discover_workloads()
        
        if not workloads:
            print_message("No workloads found", "error")
            return None
        
        workload_options = [(name, config.__doc__ or "Workload configuration") 
                           for name, config in workloads]
        
        selected = select_config_component("Workload", workload_options)
        if not selected:
            return None
        
        return next(config for name, config in workloads if name == selected)
    
    def _discover_workloads(self) -> list[tuple[str, any]]:
        """
        Dynamically discover available workload configurations.
        
        Returns:
            List of (name, config) tuples for available workloads
        """
        workloads = []
        
        try:
            import configs.workloads as workloads_module
            
            # Get all attributes that end with _config
            for name in dir(workloads_module):
                if name.endswith("_config"):
                    config = getattr(workloads_module, name)
                    workload_name = name.replace("_config", "")
                    workloads.append((workload_name, config))
            
        except Exception as e:
            print_message(f"Failed to discover workloads: {e}", "error")
        
        return workloads
    
    def _explore_config_tree(
        self,
        config     : any,
        prefix     : str,
        depth      : int = 0
    ):
        """
        Recursively explore a configuration tree.
        
        Args:
            config : The configuration object to explore
            prefix : Current path prefix for overrides
            depth  : Current recursion depth
        """
        if depth > 10:  # Prevent infinite recursion
            return
        
        # Handle different types of configurations
        if isinstance(config, dict):
            # It's a dictionary - explore each key
            components = [(k, v) for k, v in config.items() 
                         if not k.startswith("_")]
            
            if not components:
                return
            
            # Let user select which component to explore
            component_options = []
            for key, value in components:
                desc = self._get_component_description(value)
                component_options.append((key, desc))
            
            selected = select_config_component(
                f"Configuration Component (depth {depth})",
                component_options
            )
            
            if selected:
                new_prefix = f"{prefix}.{selected}" if prefix else selected
                selected_config = next(v for k, v in components if k == selected)
                self._explore_config_tree(selected_config, new_prefix, depth + 1)
        
        elif hasattr(config, "_target_"):
            # It's a hydra-zen builds object - extract the target
            target = get_target(config)
            
            if target and issubclass(target, BaseModel):
                # It's a Pydantic model - we can edit it
                self._explore_schema(target, prefix)
            else:
                # Try to get the zen_dataclass info
                if hasattr(config, "__dataclass_fields__"):
                    self._explore_dataclass(config, prefix)
                else:
                    print_message(f"Cannot explore {type(config).__name__}", "warning")
        
        else:
            print_message(f"Unknown configuration type: {type(config)}", "warning")
    
    def _get_component_description(self, component: any) -> str:
        """
        Get a description for a configuration component.
        
        Args:
            component : The component to describe
            
        Returns:
            Human-readable description
        """
        if hasattr(component, "__doc__") and component.__doc__:
            return component.__doc__.strip().split("\n")[0]
        
        if hasattr(component, "_target_"):
            target = get_target(component)
            if target:
                return f"{target.__module__}.{target.__name__}"
        
        return f"Configuration ({type(component).__name__})"
    
    def _explore_schema(
        self,
        schema_class : type[BaseModel],
        prefix       : str
    ):
        """
        Display and allow editing of Pydantic schema fields.
        
        Args:
            schema_class : The Pydantic schema class to explore
            prefix       : Prefix for Hydra overrides
        """
        print_header(f"Editing {schema_class.__name__}")
        
        if schema_class.__doc__:
            print_message(schema_class.__doc__.strip(), "info")
            console.print()
        
        # Create instance with defaults
        instance = schema_class()
        fields   = schema_class.model_fields
        
        # Build and display table
        # Create schema table
        table = Table(
            show_header  = True,
            header_style = "bold bright_cyan",
            border_style = "bright_blue",
            title        = "Schema Fields",
            show_edge    = False,
            padding      = (0, 1),
        )
        
        table.add_column("Field",       style="bright_cyan",   width=20)
        table.add_column("Type",        style="bright_white",  width=15)
        table.add_column("Default",     style="bright_yellow", width=20)
        table.add_column("Description", style="muted",         width=45)
        
        field_info = []
        for field_name, field in fields.items():
            field_type = self._get_field_type_string(field)
            default    = getattr(instance, field_name)
            desc       = field.description or ""
            
            # Truncate long descriptions
            if len(desc) > 45:
                desc = desc[:42] + "..."
            
            table.add_row(field_name, field_type, str(default), desc)
            field_info.append((field_name, field_type, default, field.description))
        
        console.print(table)
        console.print()
        
        # Interactive editing
        self._edit_fields(field_info, prefix)
        
        # Ask if user wants to continue exploring
        if console.input("[bold]Continue exploring? (y/n): [/bold]").lower() != 'y':
            return
    
    def _explore_dataclass(
        self,
        dataclass_obj : any,
        prefix        : str
    ):
        """
        Explore a hydra-zen dataclass configuration.
        
        Args:
            dataclass_obj : The dataclass object to explore
            prefix        : Prefix for Hydra overrides
        """
        print_header(f"Exploring {type(dataclass_obj).__name__}")
        
        # Create schema table
        table = Table(
            show_header  = True,
            header_style = "bold bright_cyan",
            border_style = "bright_blue",
            title        = "Dataclass Fields",
            show_edge    = False,
            padding      = (0, 1),
        )
        
        table.add_column("Field",       style="bright_cyan",   width=20)
        table.add_column("Type",        style="bright_white",  width=15)
        table.add_column("Value",       style="bright_yellow", width=20)
        table.add_column("Description", style="muted",         width=45)
        field_info = []
        
        # Get dataclass fields
        for field_name, field in dataclass_obj.__dataclass_fields__.items():
            if field_name.startswith("_"):
                continue
            
            value = getattr(dataclass_obj, field_name)
            field_type = type(value).__name__
            
            # Handle nested configurations
            if hasattr(value, "_target_") or isinstance(value, dict):
                desc = "Nested configuration (explore to edit)"
            else:
                desc = f"Value: {value}"
            
            table.add_row(field_name, field_type, str(value), desc)
            field_info.append((field_name, value))
        
        console.print(table)
        console.print()
        
        # Let user select nested configs to explore
        print_message("Enter field name to explore nested configuration", "info")
        field_to_explore = console.input("[bold]Field to explore: [/bold]").strip()
        
        if field_to_explore:
            field_data = next((f for f in field_info if f[0] == field_to_explore), None)
            if field_data:
                field_name, value = field_data
                new_prefix = f"{prefix}.{field_name}" if prefix else field_name
                self._explore_config_tree(value, new_prefix)
    
    def _edit_fields(
        self,
        field_info : list[tuple[str, str, any, str]],
        prefix     : str
    ):
        """
        Interactive field editing loop.
        
        Args:
            field_info : List of (name, type, default, description) tuples
            prefix     : Prefix for Hydra overrides
        """
        print_message("Enter field names to edit, or press Enter to finish", "info")
        
        while True:
            field_to_edit = console.input("[bold]Field to edit: [/bold]").strip()
            
            if not field_to_edit:
                break
            
            # Find the field info
            field_data = next((f for f in field_info if f[0] == field_to_edit), None)
            
            if not field_data:
                print_message(f"Field '{field_to_edit}' not found", "warning")
                continue
            
            field_name, field_type, current, description = field_data
            
            # Show full description if available
            if description and len(description) > 45:
                print_message(f"Description: {description}", "info")
            
            # Get new value
            new_value = prompt_for_field_value(field_name, field_type, str(current))
            
            if new_value is not None and str(new_value) != str(current):
                # Generate Hydra override
                override = f"{prefix}.{field_name}={new_value}" if prefix else f"{field_name}={new_value}"
                self.overrides.append(override)
                print_message(f"Added override: {override}", "success")
        
        if self.overrides:
            console.print()
            print_message(f"Generated {len(self.overrides)} configuration overrides", "success")
    
    def _get_field_type_string(self, field) -> str:
        """
        Get a readable type string for a Pydantic field.
        
        Args:
            field : Pydantic field info
            
        Returns:
            Human-readable type string
        """
        field_type = field.annotation
        
        # Handle common types
        type_map = {
            int   : "int",
            float : "float", 
            str   : "str",
            bool  : "bool",
        }
        
        if field_type in type_map:
            return type_map[field_type]
        
        if hasattr(field_type, "__origin__"):
            origin = field_type.__origin__
            args   = getattr(field_type, "__args__", ())
            
            if origin == list:
                return f"list[{self._get_simple_type_name(args[0])}]" if args else "list"
            elif origin == dict:
                return "dict"
            elif origin == tuple:
                return "tuple"
        
        return str(field_type).replace("typing.", "")
    
    def _get_simple_type_name(self, type_) -> str:
        """
        Get simple name for a type.
        
        Args:
            type_ : Type to get name for
            
        Returns:
            Simple type name
        """
        if hasattr(type_, "__name__"):
            return type_.__name__
        return str(type_)
