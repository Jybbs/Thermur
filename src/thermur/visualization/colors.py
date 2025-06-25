"""
Color mapping utilities for the Thermur visualization.

This module provides functions for creating and applying perceptually-uniform
colormaps optimized for thermal visualization. The default colormap uses a
carefully designed gradient from cool blues to hot reds that effectively
communicates temperature variations in a visually intuitive way.

The color utilities support both standard matplotlib/PyVista colormaps and
custom color gradients designed specifically for thermal data visualization.
The module includes functions for creating colormap configurations and
converting scalar values to colors with proper normalization.
"""
import matplotlib.colors as mcolors
import pyvista          as pv

from configs.schemas.visualization import ColorModel
from typing                        import Optional, Union


def create_temperature_colormap(
    color_config : Optional[ColorModel] = None,
) -> Union[str, mcolors.LinearSegmentedColormap]:
    """
    Create a temperature colormap for visualization.
    
    This function returns either a named colormap string or a custom
    LinearSegmentedColormap object optimized for thermal visualization.
    The default thermal gradient transitions smoothly from cool blues
    through greens and yellows to hot reds, following perceptual
    principles for effective temperature differentiation.
    
    The custom colormap is designed to:
    - Maximize perceptual uniformity across the temperature range
    - Provide intuitive cool-to-hot color progression
    - Maintain clarity when printed in grayscale
    - Avoid problematic color combinations for colorblind viewers
    
    Args:
        color_config : Configuration model with colormap preferences
        
    Returns:
        Either a colormap name string or custom colormap object
    """
    if color_config and color_config.colormap != "thermal":
        return color_config.colormap
    
    # Default thermal gradient colors
    thermal_colors = [
        (0.0, 0.0, 0.4),  # Dark blue
        (0.0, 0.2, 0.8),  # Blue
        (0.0, 0.5, 0.9),  # Light blue
        (0.0, 0.8, 0.8),  # Cyan
        (0.0, 0.9, 0.3),  # Green-cyan
        (0.7, 0.9, 0.0),  # Yellow-green
        (1.0, 0.8, 0.0),  # Yellow
        (1.0, 0.5, 0.0),  # Orange
        (1.0, 0.2, 0.0),  # Red
        (0.8, 0.0, 0.0),  # Dark red
    ]
    
    n_colors = len(thermal_colors)
    positions = [i / (n_colors - 1) for i in range(n_colors)]
    
    cmap_dict = {
        'red'   : [(pos, color[0], color[0]) 
                   for pos, color in zip(positions, thermal_colors)],
        'green' : [(pos, color[1], color[1]) 
                   for pos, color in zip(positions, thermal_colors)],
        'blue'  : [(pos, color[2], color[2]) 
                   for pos, color in zip(positions, thermal_colors)]
    }
    
    return mcolors.LinearSegmentedColormap('thermal', cmap_dict)


def temperature_to_color(
    temperature : float,
    max_temp    : float = 1.0,
    min_temp    : float = 0.0,
    colormap    : Optional[Union[str, mcolors.Colormap]] = None,
) -> tuple[float, float, float]:
    """
    Convert a temperature value to an RGB color.
    
    This function maps a scalar temperature value to a color tuple using
    the specified colormap and temperature range. The temperature is first
    normalized to [0, 1] based on the provided bounds, then mapped to a
    color using either a named colormap or custom thermal gradient.
    
    The normalization handles edge cases such as:
    - Temperatures outside the specified range (clamped to bounds)
    - Equal min/max temperatures (returns middle color)
    - Invalid temperature values (treated as minimum)
    
    Args:
        temperature : Temperature value to convert to color
        max_temp    : Maximum temperature for normalization
        min_temp    : Minimum temperature for normalization  
        colormap    : Colormap name or object for mapping
        
    Returns:
        RGB color tuple with values in range [0, 1]
    """
    # Handle edge case of equal bounds
    if max_temp == min_temp:
        normalized = 0.5
    else:
        normalized = (temperature - min_temp) / (max_temp - min_temp)
    
    # Clamp to valid range
    normalized = max(0.0, min(1.0, normalized))
    
    if colormap is None:
        colormap = create_temperature_colormap()
    
    if isinstance(colormap, str):
        cmap = pv.plotting.tools.get_cmap_safe(colormap)
    else:
        cmap = colormap
    
    # Extract RGB values (ignore alpha)
    return cmap(normalized)[:3]


def create_scalar_bar_config(
    color_config : Optional[ColorModel] = None,
) -> dict:
    """
    Create scalar bar configuration for temperature visualization.
    
    This function generates a configuration dictionary for PyVista's
    scalar bar (colorbar) that displays the temperature scale alongside
    the 3D visualization. The scalar bar helps users interpret the
    temperature values represented by colors in the scene.
    
    Args:
        color_config : Configuration model with scalar bar preferences
        
    Returns:
        Dictionary with scalar bar configuration parameters
    """
    default_position = (0.88, 0.25)
    default_title = "Temperature"
    
    if color_config:
        position = (
            color_config.scalar_bar_position_x,
            color_config.scalar_bar_position_y
        )
        title = color_config.scalar_bar_title
    else:
        position = default_position
        title = default_title
    
    return {
        "interactive"    : False,
        "position_x"     : position[0],
        "position_y"     : position[1],
        "title"          : title,
        "title_font_size": 14,
        "label_font_size": 12,
        "width"          : 0.08,
        "height"         : 0.4,
        "n_labels"       : 5,
        "fmt"            : "%.1f",
    }
