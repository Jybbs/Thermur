"""
System utilities for the Thermur CLI application.

This module provides functions for gathering and displaying system diagnostics,
including hardware, software, and package information. It leverages Rich to
create visually informative tables and platform-specific tools to ensure the
training environment is correctly configured.
"""
import os

from functools          import lru_cache
from importlib.metadata import PackageNotFoundError, version
from platform           import platform, python_version
from rich.console       import Console
from rich.table         import Table
from shutil             import disk_usage
from sys                import version_info
from torch              import cuda, __version__ as torch_version
from wandb              import Api, api

WANDB_API_KEY_ENV = "WANDB_API_KEY"
WANDB_ENTITY_ENV  = "WANDB_ENTITY"
WANDB_MODE_ENV    = "WANDB_MODE"


def _create_progress_bar(
    color         : str,
    used_fraction : float,
    length        : int = 20,
) -> str:
    """
    Creates a string-based progress bar using Rich markup.

    The bar visually represents a used fraction, coloring the "used" portion
    and leaving the "free" portion grey.

    Args:
        color         : The color to use for the filled portion of the bar.
        used_fraction : The fraction of the bar to fill (from 0.0 to 1.0).
        length        : The total character length of the bar.

    Returns:
        A string containing Rich markup for the progress bar.
    """
    filled   = int(used_fraction * length)
    unfilled = length - filled
    return f"[{color}]{'█' * filled}[/{color}][grey30]{'░' * unfilled}[/grey30]"


def _get_resource_details(
    available_key : str,
    info          : dict,
    missing_msg   : str,
    thresholds    : tuple[int, int],
    total_key     : str,
    total_format  : str,
) -> str:
    """
    Generates a generic, Rich-formatted string for a system resource.

    This function builds a progress bar and text for a resource (like RAM or
    disk space). The bar's color and status are determined by comparing the
    available amount against provided thresholds.

    Args:
        available_key : The key for the available/free resource in the info dict.
        info          : A dictionary containing system details.
        missing_msg   : The message to display if the resource is not found.
        thresholds    : A tuple of (low, medium) thresholds in GB.
        total_key     : The key for the total resource in the info dict.
        total_format  : The f-string precision format for the total amount.
    """
    if not info.get(total_key):
        return f"[grey50]{missing_msg}[/grey50]"

    available_gb = info[available_key]
    total_gb     = info[total_key]
    used_frac    = (total_gb - available_gb) / total_gb if total_gb > 0 else 0

    low_thresh, med_thresh = thresholds
    color = (
        "red" if available_gb < low_thresh
        else "yellow" if available_gb < med_thresh
        else "bright_green"
    )

    bar = _create_progress_bar(color=color, used_fraction=used_frac)
    return f"{bar}\n[white]{available_gb:.1f}GB free of {total_gb:{total_format}}GB[/white]"


def _safe_import(
    attr    : str,
    package : str,
) -> str | None:
    """
    Get a package attribute, handling import errors gracefully.

    Args:
        attr    : The attribute to retrieve from the package.
        package : The name of the package to import.

    Returns:
        The attribute's value, or None if import/access fails.
    """
    try:
        return getattr(__import__(package), attr)
    except (ImportError, AttributeError):
        return None


def _safe_version(
    package  : str,
    fallback : str = None,
) -> str | None:
    """
    Get a package version, handling errors gracefully.

    Args:
        package  : The name of the package to check.
        fallback : The value to return if the package is not found.

    Returns:
        The package version string, or the fallback value.
    """
    try:
        return version(package)
    except PackageNotFoundError:
        return fallback


def check_wandb_status() -> tuple[str, str]:
    """
    Check wandb installation and login status.

    Verifies whether wandb is installed and properly authenticated without
    triggering login prompts. This allows the CLI to display integration
    status without interrupting the user workflow.

    Returns:
        A tuple of (status, details) for wandb integration.
    """
    info = get_system_info()

    if not info["wandb_installed"]:
        return "[red]❌ Not Installed[/red]", "[yellow]pip install wandb[/yellow]"

    if info["wandb_user"]:
        return "[green]✅ Connected[/green]", f"[cyan]@{info['wandb_user']}[/cyan]"

    api_key_exists = os.environ.get(WANDB_API_KEY_ENV) or api.api_key
    if api_key_exists:
        return "[green]✅ API Key Set[/green]", "[white]Ready to track[/white]"

    return "[yellow]⚠️  Not Connected[/yellow]", "[yellow]Run 'wandb login'[/yellow]"


def create_system_table(console: Console) -> Table:
    """
    Create a Rich table with system information.

    Generates a comprehensive diagnostic table displaying hardware capabilities,
    software versions, and resource availability. The table uses visual indicators
    and progress bars to make the system status immediately apparent.

    Args:
        console: A Rich console instance for styling the output.

    Returns:
        A formatted Rich table containing system diagnostics.
    """
    info  = get_system_info()
    table = Table(
        border_style = "bright_blue",
        box          = None,
        header_style = "bold bright_cyan on grey15",
        padding      = (0, 1),
        show_edge    = True,
        style        = "bright_white on grey11",
        title        = "🖥️  System Diagnostics",
        title_style  = "bold bright_white on grey23",
    )
    table.add_column("Component", style="bold bright_blue", width=20)
    table.add_column("Status",    style="bold",            width=18)
    table.add_column("Details",   style="bright_white",    width=35)

    # Data-driven row definitions to reduce redundancy
    row_definitions = [

        {
            "title"   : "🔥 Thermur",
            "status"  : "[bright_green]✅ Installed[/bright_green]",
            "details" : f"[bright_cyan]v{info['thermur']}[/bright_cyan]",
        },

        {
            "title"   : "🐍 Python",
            "status"  : (
                "[bright_green]✅ Supported[/bright_green]"
                if version_info >= (3, 9)
                else "[yellow]⚠️  Outdated[/yellow]"
            ),
            "details" : f"[bright_cyan]v{info['python']}[/bright_cyan]",
        },

        {
            "title"   : "🔦 PyTorch",
            "status"  : (
                "[bright_green]✅ CUDA Ready[/bright_green]"
                if info["cuda"]
                else "[yellow]⚠️  CPU Mode[/yellow]"
            ),
            "details" : (
                f"[bright_cyan]v{info['torch']}[/bright_cyan] • "
                f"[bright_magenta]CUDA {cuda.version.cuda}[/bright_magenta]"
                if info["cuda"]
                else f"[bright_cyan]v{info['torch']}[/bright_cyan]"
            ),
        },

        {
            "title"   : "🎮 GPU",
            "status"  : (
                "[bright_green]✅ Available[/bright_green]"
                if info["cuda"]
                else "[red]❌ Not Found[/red]"
            ),
            "details" : (
                f"[bright_green]{info.get('gpu_name', '')}[/bright_green]\n"
                f"[white]Memory: {info.get('gpu_memory', 'N/A')}[/white]"
                if info["cuda"]
                else "[yellow]Training will be slower on CPU[/yellow]"
            ),
        },

        {
            "title"   : "🤖 MuJoCo",
            "status"  : (
                "[bright_green]✅ Installed[/bright_green]"
                if info["mujoco"]
                else "[red]❌ Missing[/red]"
            ),
            "details" : (
                f"[bright_cyan]v{info['mujoco']}[/bright_cyan] • Physics ready"
                if info["mujoco"]
                else "[yellow]pip install mujoco[/yellow]"
            ),
        },

        {
            "title"  : "💾 Memory",
            "status" : (
                "[grey50]❓ Unknown[/grey50]"
                if not info.get("memory_available")
                else (
                    "[red]⚠️  Low[/red]"
                    if info["memory_available"] < 4
                    else (
                        "[yellow]✅ Adequate[/yellow]"
                        if info["memory_available"] < 8
                        else "[bright_green]✅ Plenty[/bright_green]"
                    )
                )
            ),
            "details_fn" : lambda info: _get_resource_details(
                available_key = "memory_available",
                info          = info,
                missing_msg   = "Install psutil for memory info",
                thresholds    = (4, 8),
                total_key     = "memory_total",
                total_format  = ".1f",
            ),
        },

        {
            "title"  : "💿 Storage",
            "status" : (
                "[grey50]❓ Unknown[/grey50]"
                if not info.get("disk_total")
                else (
                    "[red]❌ Critical[/red]"
                    if info["disk_free"] < 5
                    else (
                        "[yellow]⚠️  Limited[/yellow]"
                        if info["disk_free"] < 20
                        else "[bright_green]✅ Available[/bright_green]"
                    )
                )
            ),
            "details_fn" : lambda info: _get_resource_details(
                available_key = "disk_free",
                info          = info,
                missing_msg   = "Could not check disk space",
                thresholds    = (5, 20),
                total_key     = "disk_total",
                total_format  = ".0f",
            ),
        },

    ]

    for row in row_definitions:
        details = row.get("details_fn", lambda _: row.get("details", ""))(info)
        table.add_row(row["title"], row["status"], details)

    return table


@lru_cache(maxsize=1)
def get_system_info() -> dict[str, any]:
    """
    Gather comprehensive system information using platform tools.

    This function collects hardware specifications, software versions, and runtime
    environment details to provide a complete picture of the training environment.
    It checks for GPU capabilities, memory availability, and the status of key
    integrations like MuJoCo and wandb.

    Returns:
        A dictionary containing system details.
    """
    cuda_available = cuda.is_available()
    info = {
        "cuda"         : cuda_available,
        "device_count" : cuda.device_count() if cuda_available else 0,
        "mujoco"       : _safe_import(attr="__version__", package="mujoco"),
        "platform"     : platform(),
        "python"       : python_version(),
        "thermur"      : _safe_version(package="thermur", fallback="dev"),
        "torch"        : torch_version,
    }

    if cuda_available:
        props              = cuda.get_device_properties(0)
        info["gpu_memory"] = f"{props.total_memory / 1e9:.1f}GB"
        info["gpu_name"]   = cuda.get_device_name(0)

    # Safely check for optional dependencies
    try:
        from psutil import virtual_memory
        mem                      = virtual_memory()
        info["memory_available"] = mem.available / 1e9
        info["memory_total"]     = mem.total / 1e9
    except ImportError:
        info["memory_available"] = 0

    try:
        usage              = disk_usage(".")
        info["disk_free"]  = usage.free / 1e9
        info["disk_total"] = usage.total / 1e9
    except Exception:
        info["disk_free"] = 0

    # Handle wandb state
    info["wandb_installed"] = _safe_import(attr="__version__", package="wandb") is not None
    info["wandb_user"]      = None
    api_key_exists          = False
    if info["wandb_installed"]:
        api_key_exists = os.environ.get(WANDB_API_KEY_ENV) or api.api_key

    if api_key_exists:
        old_mode = os.environ.get(WANDB_MODE_ENV)
        os.environ[WANDB_MODE_ENV] = "offline"

        try:
            user               = Api().viewer
            info["wandb_user"] = user.get("username") if user else None
        except Exception:
            info["wandb_user"] = None  
        finally:
            if old_mode:
                os.environ[WANDB_MODE_ENV] = old_mode
            else:
                os.environ.pop(WANDB_MODE_ENV, None)

    return info


def get_wandb_url(project: str = "thermur") -> str | None:
    """
    Generate wandb project URL if possible.

    Constructs the wandb dashboard URL based on available authentication
    information. It falls back to placeholder URLs when specific user
    information is unavailable.

    Args:
        project: The name of the wandb project.

    Returns:
        The URL to the wandb project dashboard, or None if not available.
    """
    info = get_system_info()

    if not info["wandb_installed"]:
        return None

    if info["wandb_user"]:
        return f"https://wandb.ai/{info['wandb_user']}/{project}"

    entity = os.environ.get(WANDB_ENTITY_ENV)
    return (
        f"https://wandb.ai/{entity}/{project}"
        if entity
        else f"https://wandb.ai/YOUR_USERNAME/{project}"
    )


def validate_config_overrides(overrides: list[str] | None) -> list[str]:
    """
    Validate Hydra configuration override syntax.

    Performs syntax validation on configuration overrides and checks for
    system compatibility issues. This helps catch common configuration
    errors before training begins.

    Args:
        overrides: A list of configuration overrides to validate.

    Returns:
        A list of validation issues found; empty if all are valid.
    """
    if not overrides:
        return []

    issues = []
    for o in overrides:
        if "=" not in o:
            issues.append(f"Invalid override format (missing '='): {o}")
            continue

        key            = o.split("=")[0]
        sanitized_key  = key.lstrip("+").replace(".", "").replace("_", "")
        key_is_invalid = not sanitized_key.isalnum()
        if key_is_invalid:
            issues.append(f"Invalid override key format: {o}")

    if not get_system_info()["cuda"]:
        msg = "💡 GPU not available - consider using a smaller batch size"
        issues.append(msg)

    return issues
