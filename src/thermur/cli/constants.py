"""
Constants for the thermal swarm training CLI interface.

Provides styling configurations, preset definitions, and validation parameters
for thermally-constrained drone swarm simulation and training workflows.
"""


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
            "#8B0000",      # T - dark red
            "#DC143C",      # h - crimson
            "#FF4500",      # e - orange red
            "#FF8C00",      # r - dark orange
            "#FFD700",      # m - gold
            "#FFFF00",      # u - yellow
            "#FFFACD",      # r - lemon chiffon
        ]
        
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
            },
            "standard": {
                "name"     : "standard",
                "emoji"    : "🔥",
                "desc"     : "Balanced configuration for most tasks",
                "best_for" : "Regular training runs",
            },
            "large": {
                "name"     : "large",
                "emoji"    : "💪",
                "desc"     : "High-capacity models & longer training",
                "best_for" : "Production & final models",
            },
            "debug": {
                "name"     : "debug",
                "emoji"    : "🔍",
                "desc"     : "Verbose logging & validation checks",
                "best_for" : "Troubleshooting issues",
            },
            "custom": {
                "name"     : "custom",
                "emoji"    : "🎨",
                "desc"     : "Start from scratch with full control",
                "best_for" : "Advanced users",
            },
        }
    
    
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
        
        # Resource display colors
        RESOURCE_COLOR_GOOD     = "bright_green"
        RESOURCE_COLOR_WARNING  = "yellow"
        RESOURCE_COLOR_CRITICAL = "red"
        
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
