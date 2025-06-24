"""
Color mapping utilities for the Thermur visualization.

This module provides functions for creating and applying perceptually-uniform
colormaps optimized for thermal visualization. The default colormap uses a
carefully designed gradient from cool blues to hot reds that effectively
communicates temperature variations in a visually intuitive way.
"""
import numpy as np
import pyvista as pv


def create_temperature_colormap(
    add_scalar_bar        : bool = True,
    cmap_name             : str  = "plasma",
    custom_colors         : list = None,
    reverse               : bool = False,
    scalar_bar_position_x : float = 0.88,
    scalar_bar_position_y : float = 0.25,
    scalar_bar_title      : str  = "Temperature",
) -> dict:
    """
    Create a temperature colormap configuration.
    
    This function returns a dictionary with colormap settings that can be
    used for temperature visualization in the PyVista-based rendering.
    
    The default thermal gradient is carefully designed to be both visually
    appealing and functionally effective for representing thermal data. It
    transitions from dark blue (coldest) through cyan, green, yellow, and
    finally to red (hottest), following perceptual principles to maximize
    the differentiation of temperature values.
    
    If a named colormap is preferred, any of the PyVista/Matplotlib colormaps
    can be used by specifying the cmap_name parameter. Good alternatives for
    thermal visualization include 'plasma', 'inferno', and 'viridis'.
    
    Args:
        add_scalar_bar        : Whether to add a scalar bar to the plot
        cmap_name             : Name of the colormap to use
        custom_colors         : List of custom colors if not using a named colormap
        reverse               : Whether to reverse the colormap
        scalar_bar_position_x : X-position of scalar bar (0-1)
        scalar_bar_position_y : Y-position of scalar bar (0-1)
        scalar_bar_title      : Title for the scalar bar
        
    Returns:
        Dictionary with colormap configuration
    """
    # Define a thermal-oriented color map
    if custom_colors is None:
        # Default thermal gradient: dark blue (cold) to bright red/yellow (hot)
        custom_colors = [
            (0.0, 0.0, 0.4),    # Dark blue for coldest
            (0.0, 0.2, 0.8),    # Blue
            (0.0, 0.5, 0.9),    # Light blue
            (0.0, 0.8, 0.8),    # Cyan
            (0.0, 0.9, 0.3),    # Green-cyan
            (0.7, 0.9, 0.0),    # Yellow-green
            (1.0, 0.8, 0.0),    # Yellow
            (1.0, 0.5, 0.0),    # Orange
            (1.0, 0.2, 0.0),    # Red
            (0.8, 0.0, 0.0),    # Dark red for hottest
        ]
    
    # Apply reverse if requested
    if reverse:
        custom_colors = custom_colors[::-1]
    
    return {
        "name"                : cmap_name,
        "custom_colors"       : custom_colors,
        "add_scalar_bar"      : add_scalar_bar,
        "scalar_bar_title"    : scalar_bar_title,
        "scalar_bar_position" : (scalar_bar_position_x, scalar_bar_position_y),
    }


def temperature_to_color(
    colormap    : dict = None,
    max_temp    : float = 1.0,
    min_temp    : float = 0.0,
    temperature : float = 0.5,
) -> tuple:
    """
    Convert a temperature value to a color using the specified colormap.
    
    This function maps a temperature value to a color tuple (r, g, b) using
    the provided colormap and temperature range. The temperature is first
    normalized to a 0-1 range based on the provided min and max, then mapped
    to a color using either custom color interpolation or a named colormap.
    
    The custom color interpolation uses a linear blend between the nearest
    colors in the sequence, ensuring smooth transitions. When using named
    colormaps, PyVista's built-in mapping functions are used.
    
    Args:
        colormap    : Colormap configuration from create_temperature_colormap
        max_temp    : Maximum temperature in the range
        min_temp    : Minimum temperature in the range
        temperature : Temperature value to map to color
        
    Returns:
        Tuple of (r, g, b) color values in the range [0, 1]
    """
    if colormap is None:
        colormap = create_temperature_colormap()
    
    # Normalize temperature to [0, 1] range
    if max_temp == min_temp:
        # Avoid division by zero
        normalized = 0.5
    else:
        normalized = (temperature - min_temp) / (max_temp - min_temp)
    
    # Clamp to [0, 1]
    normalized = max(0.0, min(1.0, normalized))
    
    # If using custom colors, interpolate between them
    if "custom_colors" in colormap and colormap["custom_colors"]:
        colors = colormap["custom_colors"]
        if len(colors) == 1:
            return colors[0]
        
        # Determine which color pair to interpolate between
        idx = int(normalized * (len(colors) - 1))
        if idx >= len(colors) - 1:
            return colors[-1]
        
        # Calculate fractional position between these two colors
        frac = normalized * (len(colors) - 1) - idx
        
        # Interpolate between the two colors
        color1 = colors[idx]
        color2 = colors[idx + 1]
        
        r = color1[0] * (1 - frac) + color2[0] * frac
        g = color1[1] * (1 - frac) + color2[1] * frac
        b = color1[2] * (1 - frac) + color2[2] * frac
        
        return (r, g, b)
    
    # Otherwise use named colormap from PyVista
    else:
        # Create a temporary figure with the colormap
        colormap_name = colormap.get("name", "plasma")
        cmap = pv.plotting.tools.get_cmap_safe(colormap_name)
        
        # Get the color at the normalized position
        return cmap(normalized)[:3]  # Exclude alpha channel
