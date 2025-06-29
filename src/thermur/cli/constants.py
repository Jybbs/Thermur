"""
Constants for the thermal swarm training CLI interface.

Provides styling configurations, preset definitions, and validation parameters
for thermally-constrained drone swarm simulation and training workflows.
"""
from sys import version_info


class CLIConstants:
    """
    Constants for thermal swarm training CLI operations.
    
    Namespaced access to styling, presets, and validation parameters specific
    to multi-agent thermal simulation and GNN policy training.
    """
    
    class Core:
        """
        Fundamental CLI application constants.
        
        Line length constraints, application metadata, and core configuration
        parameters governing CLI behavior.
        """
        MAX_LINE_LENGTH = 88
        APP_NAME        = "thermur"
        APP_DESCRIPTION = "🔥 Thermally-constrained drone swarm training toolkit"
    
    
    class Theme:
        """
        Thermal physics-inspired styling for CLI interface.
        
        Color mappings reflecting thermal domain physics. Fire gradient
        represents temperature transitions T_min → T_max while maintaining
        terminal readability.
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
        Message styling configurations for CLI output.
        """
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
    
    
    class Presets:
        """
        Thermal swarm training configuration presets.
        
        Parameter combinations optimized for different experimental scenarios.
        Each preset balances computational cost with training effectiveness
        for thermal constraint environments.
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
        
        Core features enabling thermal constraint modeling, multi-agent
        coordination, and imitation learning for drone swarm training.
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
        
        # Table column definitions
        TABLE_COLUMNS = [
            ("Feature",     "bright_cyan",  25, "left"),
            ("Description", "bright_white", 45, "left"),
            ("Status",      "bright_green", 12, "center"),
        ]
        
        TABLE_TITLE = "✨ Thermur Features"
    
    
    class Commands:
        """
        CLI command definitions and usage examples.
        
        Available commands for thermal swarm training workflows, including
        training execution, system validation, and experiment monitoring.
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


    class Explorer:
        """
        Constants for the interactive configuration explorer.
        
        Defines titles, messages, and table structures used to build the
        interactive exploration interface in the CLI.
        """
        # Titles and Headers
        HEADER_TITLE          = "Interactive Configuration Explorer"
        EDIT_HEADER_PREFIX    = "Editing"
        EXPLORE_HEADER_PREFIX = "Exploring"
        SCHEMA_TABLE_TITLE    = "Schema Fields"
        DATACLASS_TABLE_TITLE = "Dataclass Fields"

        # Component Names
        WORKLOAD_COMPONENT_NAME = "Workload"
        GENERIC_COMPONENT_NAME  = "Configuration Component"
        NESTED_COMPONENT_NAME   = "Nested Component"
        
        # Messages
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

        # Table Configurations
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
        
        Best practices for experiment setup, monitoring, and parameter
        optimization in thermal constraint environments.
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
        
        Environment variables and project naming conventions for experiment
        tracking in thermal swarm training workflows.
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

        # Status messages
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
        System validation requirements and diagnostics.
        
        Minimum system specifications and common configuration issues
        for thermal swarm simulation and training environments.
        """
        REQUIRED_PYTHON_VERSION = (3, 9)
        
        COMMON_ISSUES = [
            "💡 GPU not available - consider using a smaller batch size",
        ]
    
    
    class UI:
        """
        User interface constants for Rich components.
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

        # Panel settings
        PANEL_PADDING      = (1, 3)
        PANEL_BORDER_STYLE = "bright_blue"
        PANEL_BOX          = "ROUNDED"
        
        # Table settings
        TABLE_PADDING      = (0, 1)
        TABLE_TITLE_STYLE  = "bold bright_cyan"
        TABLE_HEADER_STYLE = "bold bright_blue"
        TABLE_BORDER_STYLE = "bright_blue"
        TABLE_BOX          = "MINIMAL"
        
        # Progress bar settings
        PROGRESS_BAR_WIDTH          = 30
        PROGRESS_BAR_DEFAULT_LENGTH = 20
        PROGRESS_SPINNER            = "dots"
        PROGRESS_UNFILLED_COLOR     = "grey30"
        PROGRESS_STYLE              = "thermal"
        PROGRESS_COMPLETE_STYLE     = "bright_red"
        
        # Text styling
        HEADER_TEXT_STYLE    = "bold bright_white"
        TITLE_TEXT_STYLE     = "bold bright_cyan"
        SUBTITLE_TEXT_STYLE  = "muted italic"
        COMMAND_STYLE        = "bold accent"
        MUTED_STYLE          = "muted"
        DIM_STYLE            = "dim italic"
        WHITE_STYLE          = "white"
        CYAN_STYLE           = "bright_cyan"
        SYNTAX_THEME         = "monokai"
        
        # Resource display colors
        RESOURCE_COLOR_GOOD     = "bright_green"
        RESOURCE_COLOR_WARNING  = "yellow"
        RESOURCE_COLOR_CRITICAL = "red"
        
        # A pre-formatted string for displaying system resource details.
        # It leaves placeholders for the later .format() call in ui.py.
        RESOURCE_DETAILS_TEMPLATE = (
            f"[{WHITE_STYLE}]{{:.1f}}{{}} free of {{:.1f}}{{}}[/]"
        )
        
        # Characters
        FILLED_CHAR   = "█"
        UNFILLED_CHAR = "░"
        BULLET_CHAR   = "•"
        
        # wandb
        WANDB_URL_PLACEHOLDER = "YOUR_USERNAME"
        WANDB_ICON            = "📊"
        
        # Default style fallbacks
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

        # Provides the dynamic rendering logic for each component defined in
        # SYSTEM_COMPONENTS. The UI build process will "zip" these two
        # dictionaries together using their shared keys.
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

