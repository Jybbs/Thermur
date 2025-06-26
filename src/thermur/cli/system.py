"""
System utilities for the Thermur CLI application.

This module leverages Rich's built-in system information capabilities
and platform-specific tools to provide comprehensive diagnostics.
"""
import os
import platform
import shutil
import sys

from importlib import metadata

import torch

from rich.console import Console
from rich.table   import Table


def get_system_info() -> dict[str, any]:
    """
    Gather comprehensive system information using platform tools.
    
    Returns:
        Dictionary containing system details including hardware,
        software versions, and runtime environment information
    """
    # Basic system info
    info = {
        "platform"      : platform.platform(),
        "python"        : platform.python_version(),
        "torch"         : torch.__version__,
        "cuda"          : torch.cuda.is_available(),
        "device_count"  : torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    # GPU details if available
    if info["cuda"]:
        info["gpu_name"]   = torch.cuda.get_device_name(0)
        info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
    
    # Package versions
    try:
        info["thermur"] = metadata.version("thermur")
    except metadata.PackageNotFoundError:
        info["thermur"] = "dev"
    
    # MuJoCo check
    try:
        import mujoco
        info["mujoco"] = mujoco.__version__
    except ImportError:
        info["mujoco"] = None
    
    # Memory info via psutil if available
    try:
        import psutil
        memory = psutil.virtual_memory()
        info["memory_total"]     = f"{memory.total / 1e9:.1f}GB"
        info["memory_available"] = f"{memory.available / 1e9:.1f}GB"
    except ImportError:
        info["memory_total"]     = None
        info["memory_available"] = None
    
    # wandb status - avoid triggering login prompt
    info["wandb_installed"] = False
    info["wandb_user"]      = None
    
    try:
        import wandb
        info["wandb_installed"] = True
        
        # Check if API key exists without triggering login
        api_key = os.environ.get("WANDB_API_KEY") or wandb.api.api_key
        if api_key:
            try:
                # Set offline mode temporarily to avoid login prompt
                old_mode = os.environ.get("WANDB_MODE")
                os.environ["WANDB_MODE"] = "offline"
                
                api  = wandb.Api()
                user = api.viewer
                info["wandb_user"] = user.get("username", "Unknown") if user else None
                
                # Restore mode
                if old_mode:
                    os.environ["WANDB_MODE"] = old_mode
                else:
                    os.environ.pop("WANDB_MODE", None)
            except:
                info["wandb_user"] = None
    except ImportError:
        pass
    
    return info


def create_system_table(console: Console) -> Table:
    """
    Create a Rich table with system information.
    
    Args:
        console : Rich console instance for styling
        
    Returns:
        Formatted table with system diagnostics
    """
    info = get_system_info()
    
    # Create a more visually appealing table
    table = Table(
        title    = "🖥️  System Diagnostics",
        style    = "bright_white on grey11",
        title_style = "bold bright_white on grey23",
        header_style = "bold bright_cyan on grey15",
        border_style = "bright_blue",
        show_edge = True,
        box = None,
        padding = (0, 1),
    )
    
    table.add_column("Component", style="bold bright_blue", width=20)
    table.add_column("Status",    style="bold",            width=18)
    table.add_column("Details",   style="bright_white",    width=35)
    
    # Thermur version
    table.add_row(
        "🔥 Thermur",
        "[bright_green]✅ Installed[/bright_green]",
        f"[bright_cyan]v{info['thermur']}[/bright_cyan]"
    )
    
    # Python version
    python_ok = sys.version_info >= (3, 9)
    python_status = "[bright_green]✅ Supported[/bright_green]" if python_ok else "[yellow]⚠️  Outdated[/yellow]"
    table.add_row(
        "🐍 Python",
        python_status,
        f"[bright_cyan]v{info['python']}[/bright_cyan]"
    )
    
    # PyTorch and CUDA
    cuda_status = "[bright_green]✅ CUDA Ready[/bright_green]" if info["cuda"] else "[yellow]⚠️  CPU Mode[/yellow]"
    torch_details = f"[bright_cyan]v{info['torch']}[/bright_cyan]"
    if info["cuda"]:
        torch_details += f" • [bright_magenta]CUDA {torch.version.cuda}[/bright_magenta]"
    table.add_row("🔦 PyTorch", cuda_status, torch_details)
    
    # GPU info
    if info["cuda"]:
        gpu_details = f"[bright_green]{info['gpu_name']}[/bright_green]"
        if info["gpu_memory"]:
            gpu_details += f"\n[bright_white]Memory: {info['gpu_memory']}[/bright_white]"
        table.add_row(
            "🎮 GPU",
            "[bright_green]✅ Available[/bright_green]",
            gpu_details
        )
    else:
        table.add_row(
            "🎮 GPU",
            "[red]❌ Not Found[/red]",
            "[yellow]Training will be slower on CPU[/yellow]"
        )
    
    # MuJoCo
    if info["mujoco"]:
        table.add_row(
            "🤖 MuJoCo",
            "[bright_green]✅ Installed[/bright_green]",
            f"[bright_cyan]v{info['mujoco']}[/bright_cyan] • Physics ready"
        )
    else:
        table.add_row(
            "🤖 MuJoCo",
            "[red]❌ Missing[/red]",
            "[yellow]pip install mujoco[/yellow]"
        )
    
    # Memory
    if info["memory_available"]:
        mem_gb = float(info["memory_available"].rstrip("GB"))
        total_gb = float(info["memory_total"].rstrip("GB"))
        mem_percent = (total_gb - mem_gb) / total_gb * 100
        
        if mem_gb < 4:
            mem_status = "[red]⚠️  Low Memory[/red]"
            mem_color  = "red"
        elif mem_gb < 8:
            mem_status = "[yellow]✅ Adequate[/yellow]"
            mem_color  = "yellow"
        else:
            mem_status = "[bright_green]✅ Plenty[/bright_green]"
            mem_color  = "bright_green"
        
        # Create a simple memory bar
        bar_length = 20
        filled     = int(mem_percent / 100 * bar_length)
        empty      = bar_length - filled
        mem_bar    = f"[{mem_color}]{'█' * filled}[/{mem_color}][grey30]{'░' * empty}[/grey30]"
        
        table.add_row(
            "💾 Memory",
            mem_status,
            f"{mem_bar}\n[bright_white]{info['memory_available']} free of {info['memory_total']}[/bright_white]"
        )
    else:
        table.add_row(
            "💾 Memory",
            "[grey50]❓ Unknown[/grey50]",
            "[grey50]Install psutil for memory info[/grey50]"
        )
    
    # Disk space
    try:
        _, _, free = shutil.disk_usage(".")
        total_disk = shutil.disk_usage(".").total
        free_gb    = free / 1e9
        total_gb   = total_disk / 1e9
        used_percent = (total_gb - free_gb) / total_gb * 100
        
        if free_gb < 1:
            disk_status = "[red]❌ Critical[/red]"
            disk_color  = "red"
        elif free_gb < 5:
            disk_status = "[yellow]⚠️  Limited[/yellow]"
            disk_color  = "yellow"
        else:
            disk_status = "[bright_green]✅ Available[/bright_green]"
            disk_color  = "bright_green"
        
        # Disk usage bar
        bar_length = 20
        filled     = int(used_percent / 100 * bar_length)
        empty      = bar_length - filled
        disk_bar   = f"[grey50]{'█' * filled}[/grey50][{disk_color}]{'░' * empty}[/{disk_color}]"
        
        table.add_row(
            "💿 Storage",
            disk_status,
            f"{disk_bar}\n[bright_white]{free_gb:.1f}GB free of {total_gb:.0f}GB[/bright_white]"
        )
    except:
        table.add_row(
            "💿 Storage",
            "[grey50]❓ Unknown[/grey50]",
            "[grey50]Could not check disk space[/grey50]"
        )
    
    return table


def check_wandb_status() -> tuple[str, str]:
    """
    Check wandb installation and login status.
    
    Returns:
        Tuple of (status, details) for wandb
    """
    info = get_system_info()
    
    if not info["wandb_installed"]:
        return "[red]❌ Not Installed[/red]", "[yellow]pip install wandb[/yellow]"
    
    if info["wandb_user"]:
        return "[bright_green]✅ Connected[/bright_green]", f"[bright_cyan]@{info['wandb_user']}[/bright_cyan]"
    
    # Check for API key without user info
    try:
        import wandb
        api_key = os.environ.get("WANDB_API_KEY") or wandb.api.api_key
        if api_key:
            return "[bright_green]✅ API Key Set[/bright_green]", "[bright_white]Ready to track[/bright_white]"
    except:
        pass
    
    return "[yellow]⚠️  Not Connected[/yellow]", "[yellow]Run 'wandb login'[/yellow]"


def get_wandb_url(project: str = "thermur") -> str | None:
    """
    Generate wandb project URL if possible.
    
    Args:
        project : Name of the wandb project
        
    Returns:
        URL to the wandb project or None if not available
    """
    info = get_system_info()
    
    if not info["wandb_installed"]:
        return None
    
    if info["wandb_user"]:
        return f"https://wandb.ai/{info['wandb_user']}/{project}"
    
    # Try to get from environment
    entity = os.environ.get("WANDB_ENTITY")
    if entity:
        return f"https://wandb.ai/{entity}/{project}"
    
    return f"https://wandb.ai/YOUR_USERNAME/{project}"


def validate_config_overrides(overrides: list[str] | None) -> list[str]:
    """
    Validate Hydra configuration override syntax.
    
    Args:
        overrides : List of configuration overrides to check
        
    Returns:
        List of validation issues found
    """
    if not overrides:
        return []
    
    issues = []
    
    for override in overrides:
        if '=' not in override:
            issues.append(f"Invalid override format (missing '='): {override}")
            continue
        
        key = override.split('=')[0]
        if key.startswith('+'):
            key = key[1:]
        
        # Basic validation - key should be alphanumeric with dots/underscores
        if not key.replace('.', '').replace('_', '').isalnum():
            issues.append(f"Invalid override key format: {override}")
    
    # Check GPU availability if needed
    info = get_system_info()
    if not info["cuda"]:
        issues.append("💡 GPU not available - consider using a smaller batch size")
    
    return issues
