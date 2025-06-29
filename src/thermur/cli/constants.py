"""
Constants for the thermal swarm training CLI interface.

Provides styling configurations, preset definitions, and validation parameters
for thermally-constrained drone swarm simulation and training workflows.
"""
from sys import version_info


class CLIConstants:
    """
    Provides a centralized, namespaced repository for all static CLI data.

    This class holds all static values, such as theme colors, message templates,
    preset configurations, and validation rules. It is designed to be
    instantiated a single time in the main application (`cli.py`) and then
    passed to any component that requires access to this data, ensuring
    consistency and ease of maintenance.
    """
    
    class Core:
        """
        Fundamental CLI application constants.
        
        Defines the application's top-level metadata, such as its name and a
        brief description, which are used by Typer to build the main help text.
        """
        APP_NAME        = "thermur"
        APP_DESCRIPTION = "🔥 Thermally-constrained drone swarm training toolkit"


    class Headers:
        """
        Titles and subtitles for different CLI sections.
        
        These constants are used with `ui.print_header` to create the main
        styled panels that introduce a command's function to the user.
        """
        MAIN_TITLE           = "Welcome to Thermur"
        MAIN_SUBTITLE        = "Thermal Drone Swarm Training"
        INFO_TITLE           = "Thermur System Information"
        MONITOR_TITLE        = "wandb Monitoring"
        MONITOR_SUBTITLE_TPL = "Project: {project}"
        TRAIN_TITLE          = "Thermur Training System"
        TRAIN_SUBTITLE       = "Thermally-constrained drone swarm imitation learning"
        VALIDATE_TITLE       = "System Validation"
        VALIDATE_SUBTITLE    = "Pre-flight checks for training"
        CONFIG_GEN_TITLE     = "Generated Configuration Overrides"


    class Sections:
        """
        Titles for different content sections printed to the console.
        
        These are used with `ui.print_section` to create styled horizontal
        rules that visually break up the content of a command's output.
        """
        QUICK_START         = "Quick Start"
        GETTING_STARTED     = "Getting Started"
        AVAILABLE_COMMANDS  = "Available Commands"
        INTEGRATION_STATUS  = "Integration Status"
        FEATURES            = "Features & Capabilities"
        CONFIG_SYSTEM       = "Configuration System"
        SYSTEM_VALIDATION   = "System Validation"
        CONFIG_CHECK        = "Configuration Check"
        INTEGRATION_CHECK   = "Integration Check"
        CONFIG_SETUP        = "Configuration Setup"
        INIT_TRAINING       = "Initializing Training"
        BUILDING_COMPONENTS = "Building Training Components"
        TRAINING_STARTED    = "Training Started"


    class Theme:
        """
        Thermal physics-inspired styling for CLI interface.
        
        Color mappings reflecting thermal domain physics. The fire gradient
        represents temperature transitions from T_min to T_max while
        maintaining terminal readability.
        """
        FIRE_GRADIENT = [
            "#8B0000", # T
            "#DC143C", # h
            "#FF4500", # e
            "#FF8C00", # r
            "#FFD700", # m
            "#FFFF00", # u
            "#FFFACD", # r
        ]
        
        FIRE_GRADIENT_STYLED = [f"bold {c}" for c in FIRE_GRADIENT]
        
        STYLES = {
            "info"      : "bright_cyan",
            "warning"   : "bright_yellow",
            "error"     : "bold bright_red",
            "success"   : "bold bright_green",
            "highlight" : "bold bright_blue",
            "muted"     : "grey70",
            "dim"       : "grey50",
            "thermal"   : "bold red",
            "heat"      : "bright_red on grey23",
            "drone"     : "bright_magenta",
            "swarm"     : "bright_blue",
            "accent"    : "bold bright_cyan",
        }
    
    
    class Messages:
        """
        Message styling and text configurations for CLI output.
        
        This class centralizes all user-facing strings, from simple status
        updates to formatted error messages, to ensure consistency.
        """
        SKIPPING_CHECKS          = "Skipping system checks (--force enabled)"
        TRAINING_CANCELLED       = "Training cancelled by user."
        TRAINING_INTERRUPTED     = "Training interrupted by user."
        TRAINING_FAILED_TPL      = "Training failed: {e}"
        LOADING_COMPONENTS       = "Loading training components..."
        COMPONENTS_INITIALIZED   = "All components initialized successfully!"
        MONITORING_DYNAMICS      = "Monitoring thermal constraints and swarm dynamics"
        TRACK_WANDB              = "Track progress in your wandb dashboard"
        TRAINING_COMPLETE_HEADER = "Training Complete! 🎉"
        TRAINING_COMPLETE_SUB    = "Your thermal swarm has learned to fly"
        NO_CONFIG_CHANGES        = "No configuration changes made."
        CONFIG_GEN_ADD_CMD       = "Add these to your training command:"
        CONFIG_GEN_USE_IND       = "Or use them individually:"
        WANDB_UNAVAILABLE        = "wandb not available - install with 'pip install wandb'"
        BROWSER_LAUNCH_TPL       = "Opening dashboard for project: [bright_cyan]{project}[/bright_cyan]"
        BROWSER_SUCCESS          = "Dashboard opened in your default browser!"
        BROWSER_FAIL_TPL         = "Failed to open browser: {e}"
        BROWSER_MANUAL_TPL       = "Please visit manually: {url}"
        READY_TO_TRAIN           = "Ready to train some thermal swarms? 🔥"

        TYPES = {
            "step"    : {"icon": "🔥", "style": "thermal"},
            "info"    : {"icon": "💡", "style": "info"},
            "warning" : {"icon": "🌡️", "style": "warning"},
            "error"   : {"icon": "🚨", "style": "error"},
            "success" : {"icon": "✅", "style": "success"},
            "swarm"   : {"icon": "🐦‍⬛", "style": "swarm"},
            "thermal" : {"icon": "🔥", "style": "thermal"},
            "tip"     : {"icon": "💭", "style": "muted"},
            "config"  : {"icon": "⚙️", "style": "accent"},
        }
        
        QUESTIONARY_STYLE = [
            ('question',    'fg:#ff6b6b bold'),
            ('answer',      'fg:#4ecdc4 bold'),
            ('pointer',     'fg:#ffe66d bold'),
            ('highlighted', 'fg:#ff6b6b bold'),
            ('selected',    'fg:#4ecdc4'),
            ('separator',   'fg:#95e1d3'),
            ('instruction', 'fg:#f38181'),
            ('text',        'fg:#ffffff'),
            ('disabled',    'fg:#808080 italic'),
        ]


    class Status:
        """
        Text for status indicators shown during processing.
        
        These strings are used in progress bars and spinners to provide
        real-time feedback to the user about what the application is doing.
        """
        CHECKING_REQS            = "[thermal]Checking system requirements...[/thermal]"
        VALIDATING_CONFIG        = "[accent]Validating configuration...[/accent]"
        LAUNCHING_BROWSER        = "[swarm]Launching browser...[/swarm]"
        INIT_MODULES             = "Initializing core modules..."
        LOADING_CONFIG_SYS       = "Loading configuration system..."
        REGISTERING_CONFIGS      = "Registering configurations..."
        PREPARING_HYDRA          = "Preparing Hydra runtime..."
        READY_TO_TRAIN           = "Ready to train!"
        INSTANTIATING_COMPONENTS = "Instantiating components..."
        SETUP_COMPONENT_TPL      = "Setting up {display_name}..."


    class Presets:
        """
        Thermal swarm training configuration presets.
        
        Each preset is a named collection of parameters optimized for a
        specific use case, from rapid debugging to full-scale training runs.
        """
        CONFIGS = {
            "quick": {
                "name"     : "quick",
                "emoji"    : "⚡",
                "desc"     : "Minimal setup for rapid testing",
                "best_for" : "Quick experiments & debugging",
                "prompt"   : "⚡ quick     - Fast testing & experiments",
            },
            "standard": {
                "name"     : "standard",
                "emoji"    : "🔥",
                "desc"     : "Balanced configuration for most tasks",
                "best_for" : "Regular training runs",
                "prompt"   : "🔥 standard  - Balanced performance",
            },
            "large": {
                "name"     : "large",
                "emoji"    : "💪",
                "desc"     : "High-capacity models & longer training",
                "best_for" : "Production & final models",
                "prompt"   : "💪 large     - Maximum capacity",
            },
            "debug": {
                "name"     : "debug",
                "emoji"    : "🔍",
                "desc"     : "Verbose logging & validation checks",
                "best_for" : "Troubleshooting issues",
                "prompt"   : "🔍 debug     - Detailed diagnostics",
            },
            "custom": {
                "name"     : "custom",
                "emoji"    : "🎨",
                "desc"     : "Start from scratch with full control",
                "best_for" : "Advanced users",
                "prompt"   : "🎨 custom    - Configure everything manually",
            },
        }

        TABLE_COLUMNS = [
            ("Preset",      "bright_cyan",   12, "left"),
            ("Description", "bright_white",  40, "left"),
            ("Best For",    "bright_yellow", 30, "left"),
        ]

        TABLE_TITLE = "Available Presets"
    
    
    class SystemChecks:
        """
        System validation thresholds for thermal swarm training.
        
        Resource requirements and diagnostic criteria calibrated for
        multi-agent thermal simulation and GNN policy training workloads.
        """
        MEMORY_THRESHOLDS = (4, 8)     # Low, Medium GB thresholds
        DISK_THRESHOLDS   = (5, 20)    # Low, Medium GB thresholds
        
        COMPONENTS = [
            {
                "key"         : "thermur",
                "title"       : "🔥 Thermur",
                "status_good" : "✅ Installed",
                "status_bad"  : "❌ Missing",
                "style_good"  : "bright_green",
                "style_bad"   : "red",
            },
            {
                "key"         : "python",
                "title"       : "🐍 Python",
                "status_good" : "✅ Supported",
                "status_bad"  : "⚠️  Outdated",
                "style_good"  : "bright_green",
                "style_bad"   : "yellow",
            },
            {
                "key"         : "torch",
                "title"       : "🔦 PyTorch",
                "status_good" : "✅ CUDA Ready",
                "status_bad"  : "⚠️  CPU Mode",
                "style_good"  : "bright_green",
                "style_bad"   : "yellow",
            },
            {
                "key"         : "gpu",
                "title"       : "🎮 GPU",
                "status_good" : "✅ Available",
                "status_bad"  : "❌ Not Found",
                "style_good"  : "bright_green",
                "style_bad"   : "red",
            },
            {
                "key"         : "mujoco",
                "title"       : "🤖 MuJoCo",
                "status_good" : "✅ Installed",
                "status_bad"  : "❌ Missing",
                "style_good"  : "bright_green",
                "style_bad"   : "red",
            },
            {
                "key"         : "memory",
                "title"       : "💾 Memory",
                "status_good" : "✅ Plenty",
                "status_ok"   : "✅ Adequate",
                "status_bad"  : "⚠️  Low",
                "style_good"  : "bright_green",
                "style_ok"    : "yellow",
                "style_bad"   : "red",
            },
            {
                "key"         : "disk",
                "title"       : "💿 Storage",
                "status_good" : "✅ Available",
                "status_ok"   : "⚠️  Limited",
                "status_bad"  : "❌ Critical",
                "style_good"  : "bright_green",
                "style_ok"    : "yellow",
                "style_bad"   : "red",
            },
        ]
    
    
    class Features:
        """
        Thermur platform capabilities and implementation status.
        
        This list defines the core features of the platform, which are
        displayed in a table by the `info` command.
        """
        LIST = [
            {
                "name"   : "🔥 Thermal Constraints",
                "desc"   : "Realistic heat modeling for drone swarms",
                "status" : "✅ Ready",
            },
            {
                "name"   : "🐦‍⬛ Swarm Intelligence",
                "desc"   : "Multi-agent coordination and flocking",
                "status" : "✅ Ready",
            },
            {
                "name"   : "🎓 Imitation Learning",
                "desc"   : "Learn from expert demonstrations",
                "status" : "✅ Ready",
            },
            {
                "name"   : "📊 wandb Integration",
                "desc"   : "Real-time experiment tracking",
                "status" : "✅ Ready",
            },
            {
                "name"   : "🎮 GPU Acceleration",
                "desc"   : "CUDA support for fast training",
                "status" : "✅ Ready",
            },
            {
                "name"   : "🔧 Hydra Configuration",
                "desc"   : "Flexible experiment configuration",
                "status" : "✅ Ready",
            },
            {
                "name"   : "📈 Live Visualization",
                "desc"   : "Real-time swarm behavior rendering",
                "status" : "🚧 Beta",
            },
        ]
        
        TABLE_COLUMNS = [
            ("Feature",     "bright_cyan",  25, "left"),
            ("Description", "bright_white", 45, "left"),
            ("Status",      "bright_green", 12, "center"),
        ]
        
        TABLE_TITLE = "✨ Thermur Features"
    
    
    class Commands:
        """
        CLI command definitions and usage examples.
        
        This class holds the metadata for all available CLI commands, which is
        used to dynamically generate help text and welcome screens.
        """
        AVAILABLE = [
            {
                "name" : "train",
                "icon" : "🔥",
                "desc" : "Train a thermal drone swarm with imitation learning",
            },
            {
                "name" : "configure",
                "icon" : "🔧",
                "desc" : "Interactively explore and edit configurations",
            },
            {
                "name" : "info",
                "icon" : "📊",
                "desc" : "Display system information and capabilities",
            },
            {
                "name" : "validate",
                "icon" : "✅",
                "desc" : "Validate system setup and configuration",
            },
            {
                "name" : "monitor",
                "icon" : "📈",
                "desc" : "Open wandb dashboard to monitor experiments",
            },
        ]
        
        EXAMPLES = [
            {
                "desc"    : "Start your first training run",
                "command" : "thermur train --preset quick",
                "note"    : "Perfect for testing the system",
            },
            {
                "desc"    : "Check your system setup",
                "command" : "thermur info",
                "note"    : "See what's installed and ready",
            },
            {
                "desc"    : "Get help on any command",
                "command" : "thermur train --help",
                "note"    : "Detailed usage information",
            },
        ]

        OVERRIDE_SYNTAX_HELP = (
            "# Override examples:\n"
            "hyperparameters.lr=0.001          # Learning rate\n"
            "hyperparameters.batch_size=64     # Batch size\n"
            "swarm.num_drones=10               # Number of drones\n"
            "environment.max_temp=85.0         # Temperature limit\n"
            "+experiment=my_custom_setup       # Load experiment"
        )

        OVERRIDE_SYNTAX_TITLE = "Configuration Override Syntax"


    class Training:
        """
        Configuration for the training process and component instantiation.
        
        This class decouples the `cli.py` orchestrator from the specific set
        of components required for a training run.
        """
        COMPONENT_CONFIGS = [
            ("environment",       "simulation",        "🌍 Environment"),
            ("expert_policy",     "expert_policy",     "🎓 Expert Policy"),
            ("policy",            "policy",            "🧠 Learning Policy"),
            ("data_collector",    "data_collector",    "📊 Data Collector"),
            ("experience_buffer", "experience_buffer", "💾 Experience Buffer"),
            ("loss_function",     "loss_function",     "📏 Loss Function"),
            ("optimizer",         "optimizer",         "⚙️  Optimizer"),
            ("hyperparameters",   "hyperparameters",   "🎛️  Hyperparameters"),
            ("wandb_config",      "monitoring.wandb",  "📊 wandb Tracking"),
        ]

        VISUALIZER_KEY = "visualization"


    class Explorer:
        """
        Constants for the interactive configuration explorer.
        
        Defines titles, messages, and table structures used to build the
        interactive exploration interface in the CLI.
        """
        HEADER_TITLE          = "Interactive Configuration Explorer"
        EDIT_HEADER_PREFIX    = "Editing"
        EXPLORE_HEADER_PREFIX = "Exploring"
        SCHEMA_TABLE_TITLE    = "Schema Fields"
        DATACLASS_TABLE_TITLE = "Dataclass Fields"

        WORKLOAD_COMPONENT_NAME = "Workload"
        GENERIC_COMPONENT_NAME  = "Configuration Component"
        NESTED_COMPONENT_NAME   = "Nested Component"
        
        PROMPT_TO_EDIT       = "Enter field names to edit, or press Enter to finish."
        FIELD_PROMPT         = "[bold]Field to edit: [/bold]"
        NO_WORKLOADS_FOUND   = "No workload configurations found"
        CONFIG_IMPORT_FAILED = "Failed to import configs module"
        MAX_DEPTH_REACHED    = "Reached maximum exploration depth"
        FIELD_NOT_FOUND      = "Field '{field_name}' not found"
        UNKNOWN_CONFIG_TYPE  = "Unknown configuration type: {type_name}"
        CANNOT_EXPLORE       = "Cannot explore target: {type_name}"
        OVERRIDE_ADDED       = "Added override: {override}"
        OVERRIDES_GENERATED  = "Generated {count} configuration overrides"
        DEFAULT_WORKLOAD_DOC = "Workload configuration"

        SCHEMA_TABLE_COLUMNS = [
            ("Field",       "bright_cyan",   20, "left"),
            ("Type",        "bright_white",  15, "left"),
            ("Default",     "bright_yellow", 20, "left"),
            ("Description", "muted",         45, "left"),
        ]
        
        DATACLASS_TABLE_COLUMNS = [
            ("Field",          "bright_cyan",   20, "left"),
            ("Type",           "bright_white",  20, "left"),
            ("Value / Target", "bright_yellow", 45, "left"),
        ]
    
    
    class Tips:
        """
        Thermal swarm training workflow guidance.
        
        These tips are displayed by the `info` command to provide users with
        quick, actionable advice for using the CLI effectively.
        """
        TRAINING = [
            {
                "desc"    : "Use presets for quick starts",
                "command" : "thermur train --preset quick",
            },
            {
                "desc"    : "Monitor training live",
                "command" : "thermur monitor --project my-experiment",
            },
            {
                "desc"    : "Explore configurations",
                "command" : "thermur configure",
            },
            {
                "desc"    : "Override any parameter",
                "command" : "thermur train --config hyperparameters.lr=0.001",
            },
        ]
        
        SECTION_TITLE = "💡 Quick Tips"
        SECTION_STYLE = "bright_yellow"
        BULLET_STYLE  = "bright_yellow"
    
    
    class Wandb:
        """
        Weights & Biases integration configuration.
        
        This class centralizes environment variable keys and project naming
        conventions for experiment tracking in wandb.
        """
        API_KEY_ENV = "WANDB_API_KEY"
        ENTITY_ENV  = "WANDB_ENTITY"
        MODE_ENV    = "WANDB_MODE"
        
        DEFAULT_PROJECT = "thermur"
        
        EXAMPLE_PROJECTS = [
            "thermal-swarm-v1",
            "drone-flocking-experiments",
            "heat-aware-navigation",
            "imitation-learning-tests",
        ]

        STATUS_NOT_INSTALLED  = "[red]❌ Not Installed[/red]"
        DETAILS_NOT_INSTALLED = "[yellow]pip install wandb[/yellow]"
        STATUS_CONNECTED      = "[green]✅ Connected[/green]"
        DETAILS_CONNECTED     = "[cyan]@{user}[/cyan]"
        STATUS_API_KEY        = "[green]✅ API Key Set[/green]"
        DETAILS_API_KEY       = "[white]Ready to track[/white]"
        STATUS_NOT_CONNECTED  = "[yellow]⚠️  Not Connected[/yellow]"
        DETAILS_NOT_CONNECTED = "[yellow]Run 'wandb login'[/yellow]"
    
    
    class Validation:
        """
        System validation requirements and diagnostic messages.
        
        This class holds minimum system specifications and common configuration
        issues for thermal swarm simulation and training environments.
        """
        REQUIRED_PYTHON_VERSION = (3, 9)
        
        CONFIG_FAIL_MSG          = "Configuration validation failed:"
        CONFIG_ISSUES_FOUND      = "Configuration issues found:"
        CONFIG_VALIDATION_PASSED = "Configuration validation passed!"
        VALIDATION_WITH_WARNINGS = "⚠️  Validation completed with warnings"
        REVIEW_ISSUES_TIP        = "Review the issues above before training"
        ALL_VALIDATIONS_PASSED   = "✅ All validations passed!"
        SYSTEM_READY             = "Your system is ready for training"
        FORCE_OVERRIDE_TIP       = "Use --force to override or fix the issues above."
    
    
    class UI:
        """
        User interface constants for Rich components.
        
        Defines static configuration for all Rich-rendered components, such
        as padding, border styles, colors, and character sets.
        """
        CATEGORY_EMOJIS = {
            "hyperparameters" : "🎛️",
            "environment"     : "🌍",
            "swarm"           : "🐦‍⬛",
            "policy"          : "🧠",
            "monitoring"      : "📊",
            "visualization"   : "📈",
            "default"         : "⚙️",
        }

        PANEL_PADDING      = (1, 3)
        PANEL_BORDER_STYLE = "bright_blue"
        PANEL_BOX          = "ROUNDED"
        
        TABLE_PADDING      = (0, 1)
        TABLE_TITLE_STYLE  = "bold bright_cyan"
        TABLE_HEADER_STYLE = "bold bright_blue"
        TABLE_BORDER_STYLE = "bright_blue"
        TABLE_BOX          = "MINIMAL"
        
        PROGRESS_BAR_WIDTH          = 30
        PROGRESS_BAR_DEFAULT_LENGTH = 20
        PROGRESS_SPINNER            = "dots"
        PROGRESS_UNFILLED_COLOR     = "grey30"
        PROGRESS_STYLE              = "thermal"
        PROGRESS_COMPLETE_STYLE     = "bright_red"
        
        HEADER_TEXT_STYLE   = "bold bright_white"
        TITLE_TEXT_STYLE    = "bold bright_cyan"
        SUBTITLE_TEXT_STYLE = "muted italic"
        COMMAND_STYLE       = "bold accent"
        MUTED_STYLE         = "muted"
        DIM_STYLE           = "dim italic"
        WHITE_STYLE         = "white"
        CYAN_STYLE          = "bright_cyan"
        SYNTAX_THEME        = "monokai"
        
        RESOURCE_COLOR_GOOD     = "bright_green"
        RESOURCE_COLOR_WARNING  = "yellow"
        RESOURCE_COLOR_CRITICAL = "red"
        
        RESOURCE_DETAILS_TEMPLATE = (
            f"[{WHITE_STYLE}]{{:.1f}}{{}} free of {{:.1f}}{{}}[/]"
        )
        
        FILLED_CHAR   = "█"
        UNFILLED_CHAR = "░"
        BULLET_CHAR   = "•"
        
        WANDB_URL_PLACEHOLDER = "YOUR_USERNAME"
        WANDB_ICON            = "📊"
        
        DEFAULT_SECTION_STYLE = "accent"
        DEFAULT_BADGE_STYLE   = "success"


    class System:
        """
        Constants for system diagnostics, resource checking, and validation.

        This class holds static data for generating system diagnostic tables,
        defining resource thresholds, and providing standardized validation
        messages used throughout the CLI.
        """
        GPU_UNAVAILABLE         = "💡 GPU not available - consider using a smaller batch size"
        INVALID_OVERRIDE_FORMAT = "Invalid override format (missing '=')"
        INVALID_OVERRIDE_KEY    = "Invalid override key format"
        PROGRESS_BAR_LENGTH     = 20
        RESOURCE_MISSING_MSG    = {
            "disk"   : "Could not check disk space",
            "memory" : "Install psutil for memory info",
        }

        TABLE_COLUMNS = [
            {
                "header" : "Component",
                "style"  : "bold bright_blue",
                "width"  : 20
            },
            {
                "header" : "Status",
                "style"  : "bold",
                "width"  : 18
            },
            {
                "header" : "Details",
                "style"  : "bright_white",
                "width"  : 35
            },
        ]

        SYSTEM_COMPONENTS = {
            "thermur" : "🔥 Thermur",
            "python"  : "🐍 Python",
            "torch"   : "🔦 PyTorch",
            "gpu"     : "🎮 GPU",
            "mujoco"  : "🤖 MuJoCo",
            "memory"  : "💾 Memory",
            "storage" : "💿 Storage",
        }

        SYSTEM_LOGIC = {

            "thermur": {
                "status"  : lambda info: "[bright_green]✅ Installed[/bright_green]",
                "details" : lambda info: f"[bright_cyan]v{info['thermur']}[/bright_cyan]",
            },

            "python": {
                "status"  : lambda info: (
                    "[bright_green]✅ Supported[/bright_green]"
                    if info["python_version_info"] >= (3, 9)
                        else "[yellow]⚠️  Outdated[/yellow]"
                ),
                "details" : lambda info: f"[bright_cyan]v{info['python']}[/bright_cyan]",
            },

            "torch": {
                "status"  : lambda info: (
                    "[bright_green]✅ CUDA Ready[/bright_green]"
                    if info["cuda"]
                        else "[yellow]⚠️  CPU Mode[/yellow]"
                ),
                "details" : lambda info: (
                    f"[bright_cyan]v{info['torch']}[/bright_cyan] • "
                    f"[bright_magenta]CUDA {info['cuda_version']}[/bright_magenta]"
                    if info["cuda"]
                        else f"[bright_cyan]v{info['torch']}[/bright_cyan]"
                ),
            },

            "gpu": {
                "status"  : lambda info: (
                    "[bright_green]✅ Available[/bright_green]"
                    if info["cuda"]
                        else "[red]❌ Not Found[/red]"
                ),
                "details" : lambda info: (
                    f"[bright_green]{info.get('gpu_name', '')}[/bright_green]\n"
                    f"[white]Memory: {info.get('gpu_memory', 'N/A')}[/white]"
                    if info["cuda"]
                        else "[yellow]Training will be slower on CPU[/yellow]"
                ),
            },

            "mujoco": {
                "status"  : lambda info: (
                    "[bright_green]✅ Installed[/bright_green]"
                    if info["mujoco"]
                        else "[red]❌ Missing[/red]"
                ),
                "details" : lambda info: (
                    f"[bright_cyan]v{info['mujoco']}[/bright_cyan] • Physics ready"
                    if info["mujoco"]
                        else "[yellow]pip install mujoco[/yellow]"
                ),
            },

            "memory": {
                "is_resource" : True,
                "status"      : lambda info: (
                    "[grey50]❓ Unknown[/grey50]"
                    if not info.get("memory_total")
                        else "[red]⚠️  Low[/red]"           if info["memory_available"] < 4
                        else "[yellow]✅ Adequate[/yellow]" if info["memory_available"] < 8
                        else "[bright_green]✅ Plenty[/bright_green]"
                ),
            },

            "storage": {
                "is_resource" : True,
                "status"      : lambda info: (
                    "[grey50]❓ Unknown[/grey50]"
                    if not info.get("disk_total")
                        else "[red]❌ Critical[/red]"       if info["disk_free"] < 5
                        else "[yellow]⚠️  Limited[/yellow]" if info["disk_free"] < 20
                        else "[bright_green]✅ Available[/bright_green]"
                ),
            },
        }

        TABLE_SETTINGS = {
            "border_style" : "bright_blue",
            "box"          : None,
            "header_style" : "bold bright_cyan on grey15",
            "padding"      : (0, 1),
            "show_edge"    : True,
            "style"        : "bright_white on grey11",
            "title_style"  : "bold bright_white on grey23",
        }
        TABLE_TITLE = "🖥️  System Diagnostics"