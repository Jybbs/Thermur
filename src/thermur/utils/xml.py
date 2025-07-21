"""
MuJoCo XML generation utilities for the Thermur project.

This module provides functions for dynamically generating and loading MuJoCo
models with varying numbers of agents, enabling true multi-agent physics
simulation for flock environments.
"""
from pathlib import Path

import mujoco as mj


def generate_flock_xml(
    assets_dir      : Path,
    shape           : tuple[int, int],
    simulation_step : float
) -> str:
    """
    Dynamically generates a MuJoCo XML model with N distinct drone bodies.
    
    This function creates a complete XML model by:
    1. Reading the flock.xml template
    2. Reading the drone.xml template for each agent
    3. Replacing template variables with appropriate values for each agent
    4. Inserting all N drone bodies and their actuators into the flock model
    
    Args:
        assets_dir      : Path to the directory containing XML templates
        shape           : (agent_count, spatial_dims) tuple from FlockModel
        simulation_step : Physics simulation timestep in seconds
    
    Returns:
        A string containing the complete MuJoCo XML model
    """
    n, dims        = shape
    drone_template = (assets_dir / "drone.xml").read_text()
    flock_template = (assets_dir / "flock.xml").read_text()
    actuator_defs  = []
    drone_bodies   = []
    
    for i in range(n):
        # Create a slight offset for each drone to prevent initial collisions
        offset   = 0.3 * i
        position = f"0 {offset} 0" if dims == 3 else f"0 {offset}"
        
        joints_xml = ""
        for j in range(dims):
            axis = ["1 0 0", "0 1 0", "0 0 1"][j]
            joints_xml += f"""
            <joint
                axis    = "{axis}"
                limited = "true"
                name    = "drone_{i}_joint_{j}"
                pos     = "0 0 0"
                range   = "-10 10"
                type    = "slide"
            />
            """
        
        drone_body = (
            drone_template
            .replace("$AGENT_ID$", str(i))
            .replace("$POSITION$", position)
            .replace("<!-- JOINTS_XML -->", joints_xml)
        )
        
        drone_bodies.append(drone_body)
        
        for j in range(dims):
            axis_name = ["x", "y", "z"][j]
            actuator_defs.append(f"""
        <velocity
            ctrllimited = "true"
            ctrlrange   = "-1 1"
            gear        = "1"
            joint       = "drone_{i}_joint_{j}"
            name        = "drone_{i}_vel_{axis_name}"
        />""")
    
    flock_xml = (
        flock_template
        .replace("$TIMESTEP$", str(simulation_step))
        .replace("<!-- DRONE_BODIES -->", "\n".join(drone_bodies))
        .replace("<!-- ACTUATORS -->", "\n".join(actuator_defs))
    )
    
    return flock_xml


def load_flock_model(xml_string: str) -> dict:
    """
    Loads a MuJoCo model from the provided XML string.
    
    This function creates MuJoCo model and data objects from the XML string,
    which can then be used for physics simulation.
    
    Args:
        xml_string: A string containing the MuJoCo XML model
    
    Returns:
        A dictionary containing the MuJoCo model and data objects
    """
    model = mj.MjModel.from_xml_string(xml_string)
    data  = mj.MjData(model)
    
    return {
        "data"  : data,
        "model" : model
    }
