"""
Interactive configuration explorer for the Thermur CLI.

This module provides the ConfigExplorer class, which allows users to
interactively browse and edit hydra-zen configuration schemas. The primary
goal is to offer a guided way to discover available parameters and generate
valid Hydra override strings for a training run.
"""
from hydra_zen  import get_target
from omegaconf  import DictConfig
from pydantic   import BaseModel


class ConfigExplorer:
    """
    Interactively discovers, navigates, and edits configuration schemas.

    This class orchestrates the user-facing workflow for exploring nested
    configuration objects. It uses `ThermurUI` for rendering, `CLIPrompts` for
    user input, and DictConfig for static text and settings, all of which are injected
    during initialization. This keeps the explorer focused on the logic of
    navigating the config tree while delegating I/O and data to other components.
    """

    def __init__(
        self, 
        ui, 
        prompts, 
        explorer_config: DictConfig
    ):
        """
        Initializes the configuration explorer with its dependencies.

        Args:
            ui              : An initialized `ThermurUI` object for rendering components.
            prompts         : An initialized `CLIPrompts` object for handling user input.
            explorer_config : Explorer configuration from DictConfig.
        """
        self.overrides = []
        self.ui        = ui
        self.prompts   = prompts
        self.config    = explorer_config
    
    def explore_interactive(self) -> list[str]:
        """
        Launch the interactive configuration exploration and editing session.
        
        This is the main entry point for the explorer. It guides the user
        through selecting a workload, Browse its components, and editing
        parameters, finally returning any generated overrides.
        
        Returns:
            A list of Hydra configuration override strings.
        """
        self.ui.print_header(self.config.header_title)
        
        # Import heavy modules only when the explorer is used
        try:
            from configs.imitation.workloads import imitation_config
            from configs.cli.workloads import cli_config
        except ImportError:
            self.ui.print_message(
                self.config.config_import_failed, "error"
            )
            return []
        
        workload_config = self._select_workload()
        if not workload_config:
            return []
        
        self._explore_config_tree(workload_config, prefix="")
        
        if self.overrides:
            self.ui.console.print()
            self.ui.print_message(
                self.config.overrides_generated.format(
                    count=len(self.overrides)
                ), 
                "success"
            )
            
        return self.overrides
    
    def _select_workload(self) -> dict | None:
        """
        Discover and prompt the user to select a workload configuration.
        
        This method finds all available configurations and presents them to
        the user for selection before starting the exploration process.
        
        Returns:
            The selected workload configuration object, or None if cancelled.
        """
        workloads = self._discover_workloads()
        if not workloads:
            self.ui.print_message(
                self.config.no_workloads_found, "error"
            )
            return None
        
        workload_options = [
            (name, doc or self.config.default_workload_doc) 
            for name, doc in workloads
        ]
        
        selected_name = self.prompts.select_config_component(
            self.config.workload_component_name, workload_options
        )
        return next(
            (config for name, config in workloads if name == selected_name), 
            None
        )
    
    def _discover_workloads(self) -> list[tuple[str, any]]:
        """
        Dynamically discover available workload configurations.
        
        It inspects the `configs.workloads` module for any objects that follow
        the `_config` naming convention.
        
        Returns:
            A list of (name, config_object) tuples for available workloads.
        """
        try:
            # Import to keep CLI startup fast
            from configs.imitation import imitation_config
            from configs.cli       import cli_config
            
            return [
                ("imitation", imitation_config),
                ("cli",       cli_config)
            ]

        except Exception as e:
            msg = f"{self.config.no_workloads_found}: {e}"
            self.ui.print_message(msg, "error")
            return []
    
    def _explore_config_tree(
        self, 
        config : any, 
        prefix : str, 
        depth  : int = 0
    ):
        """
        Recursively explore a configuration tree node.

        This method acts as a router, dispatching to the correct handler based
        on the type of the configuration node (e.g., dict, Pydantic model).
        
        Args:
            config : The configuration object or dictionary to explore.
            prefix : The current dot-path prefix for generating overrides.
            depth  : The current recursion depth to prevent infinite loops.
        """
        if depth > 10:
            self.ui.print_message(
                self.config.max_depth_reached, "warning"
            )
            return

        if isinstance(config, dict):
            self._explore_dict_node(config, prefix, depth)
        elif hasattr(config, "_target_"):
            self._explore_builds_node(config, prefix, depth)
        else:
            self.ui.print_message(
                self.config.unknown_config_type.format(
                    type_name=type(config).__name__
                ), 
                "warning"
            )

    def _explore_dict_node(
        self, 
        config_dict : dict, 
        prefix      : str, 
        depth       : int
    ):
        """
        Handle the exploration of a dictionary node in the config tree.
        
        Args:
            config_dict : The dictionary to explore.
            prefix      : The current dot-path prefix.
            depth       : The current recursion depth.
        """
        component_options = [
            (key, self.ui.get_component_description(value))
            for key, value in config_dict.items() if not key.startswith("_")
        ]

        if not component_options:
            return

        title        = f"{self.config.generic_component_name} (depth {depth})"
        selected_key = self.prompts.select_config_component(title, component_options)
        
        if selected_key:
            new_prefix = f"{prefix}.{selected_key}" if prefix else selected_key
            selected_config = config_dict[selected_key]
            self._explore_config_tree(selected_config, new_prefix, depth + 1)

    def _explore_builds_node(
        self, 
        config_builds : any, 
        prefix        : str, 
        depth         : int
    ):
        """
        Handle the exploration of a hydra-zen `builds` object node.
        
        Args:
            config_builds : The `builds` object to explore.
            prefix        : The current dot-path prefix.
            depth         : The current recursion depth.
        """
        target = get_target(config_builds)

        if target and issubclass(target, BaseModel):
            self._explore_pydantic_schema(target, prefix)
        elif hasattr(config_builds, "__dataclass_fields__"):
            self._explore_dataclass(config_builds, prefix, depth)
        else:
            self.ui.print_message(
                self.config.cannot_explore.format(
                    type_name=type(target).__name__
                ), 
                "warning"
            )
    
    def _explore_pydantic_schema(
        self, 
        schema_class : type[BaseModel], 
        prefix       : str
    ):
        """
        Display and allow editing of a Pydantic model's schema fields.
        
        Args:
            schema_class : The Pydantic model class to explore.
            prefix       : The dot-path prefix for Hydra overrides.
        """
        header = (
            f"{self.config.edit_header_prefix} {schema_class.__name__}"
        )
        self.ui.print_header(header)
        
        if doc := getattr(schema_class, "__doc__", None):
            self.ui.print_message(doc.strip(), "info")
            self.ui.console.print()

        instance      = schema_class()
        schema_fields = schema_class.model_fields
        field_info    = []
        
        table = self.ui.create_aligned_table(
            title   = self.config.schema_table_title,
            columns = self.config.schema_table_columns,
        )
        
        for name, field in schema_fields.items():
            description = field.description or ""
            field_type  = self.ui.get_field_type_string(field.annotation)
            default_val = getattr(instance, name)
            
            field_info.append((name, field_type, default_val, description))
            
            desc_str = (
                (description[:42] + "...") if len(description) > 45 else description
            )
            table.add_row(name, field_type, str(default_val), desc_str)
        
        self.ui.console.print(table)
        self.ui.console.print()
        
        new_overrides = self.prompts.edit_multiple_fields(
            fields      = field_info,
            prefix      = prefix,
            description = self.config.prompt_to_edit
        )
        self.overrides.extend(new_overrides)
    
    def _explore_dataclass(
        self, 
        dataclass_obj : any, 
        prefix        : str, 
        depth         : int
    ):
        """
        Display and allow navigation of a hydra-zen dataclass configuration.
        
        Args:
            dataclass_obj : The dataclass object to explore.
            prefix        : The dot-path prefix for Hydra overrides.
            depth         : The current recursion depth.
        """
        header = (
            f"{self.config.explore_header_prefix} "
            f"{type(dataclass_obj).__name__}"
        )
        self.ui.print_header(header)
        
        table = self.ui.create_aligned_table(
            title   = self.config.dataclass_table_title,
            columns = self.config.dataclass_table_columns,
        )
        
        field_info = [
            (name, getattr(dataclass_obj, name))
            for name in dataclass_obj.__dataclass_fields__ 
            if not name.startswith("_")
        ]
        
        for name, value in field_info:
            description = self.ui.get_component_description(value)
            table.add_row(name, type(value).__name__, description)
            
        self.ui.console.print(table)
        self.ui.console.print()

        component_options = [
            (name, self.ui.get_component_description(val)) 
            for name, val in field_info
        ]
        field_to_explore = self.prompts.select_config_component(
            self.config.nested_component_name, component_options
        )
        
        if field_to_explore:
            selected_value = next(
                (val for name, val in field_info if name == field_to_explore), 
                None
            )

            if selected_value:
                new_prefix = (
                    f"{prefix}.{field_to_explore}" 
                    if prefix 
                    else field_to_explore
                )
                self._explore_config_tree(selected_value, new_prefix, depth + 1)