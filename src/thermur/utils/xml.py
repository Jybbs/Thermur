"""
Utilities for dynamic model generation.

This module provides functions for dynamically generating MuJoCo XML models
for multi-agent physics simulations. It uses component-based templates
to assemble complete models with varying numbers of agents and properties.
"""
import mujoco

from pathlib import Path


def generate_swarm_xml(
    assets_dir      : Path,
    agent_count     : int,
    spatial_dims    : int,
    simulation_step : float
) -> str:
    """
    Generates a complete MuJoCo XML model for a swarm of agents.
    
    This function assembles a MuJoCo XML model by reading component templates
    from the assets directory and substituting the appropriate values for
    agent count, dimensions, and simulation parameters. The resulting model
    contains N distinct, independently controllable agents.
    
    Args:
        assets_dir      : Path to directory containing XML component templates
        agent_count     : Number of agents to include in the model
        spatial_dims    : Number of spatial dimensions (2 or 3)
        simulation_step : Time step for physics simulation in seconds
        
    Returns:
        Complete MuJoCo XML model as a string
    """
    drone_template = (assets_dir / "drone.xml").read_text()
    swarm_template = (assets_dir / "swarm.xml").read_text()
    
    dim_names = ["x", "y", "z"][:spatial_dims]
    axes      = ["1 0 0", "0 1 0", "0 0 1"][:spatial_dims]
    
    drone_bodies = []
    for i in range(agent_count):
        position = f"{i * 0.5} 0" if spatial_dims == 2 else f"{i * 0.5} 0 0"
        joints   = [
            f"""            <joint 
                axis    = "{axes[d]}" 
                limited = "false"
                name    = "drone_{i}_joint_{dim_names[d]}" 
                type    = "slide"
            />""" 
            for d in range(spatial_dims)
        ]
        
        # Create drone body with substituted values
        drone_xml = drone_template.replace("$AGENT_ID$", str(i))
        drone_xml = drone_xml.replace("$POSITION$", position)
        drone_xml = drone_xml.replace("<!-- JOINTS_XML -->", "\n".join(joints))
        
        drone_bodies.append(drone_xml)
    
    actuators = [
        f"""        <velocity 
            ctrlrange = "-10 10" 
            joint     = "drone_{i}_joint_{dim_names[d]}" 
            kv        = "100"
            name      = "vel_{i}_{dim_names[d]}"
        />"""
        for i in range(agent_count)
        for d in range(spatial_dims)
    ]
    
    model_xml = (
        swarm_template
            .replace("$TIMESTEP$", str(simulation_step))
            .replace("<!-- DRONE_BODIES -->", "\n".join(drone_bodies))
            .replace("<!-- ACTUATORS -->", "\n".join(actuators))
    )
    
    return model_xml


def load_swarm_model(
    xml_string : str,
    timestep   : float = None
) -> dict:
    """
    Loads a MuJoCo model from an XML string and configures it.
    
    Args:
        xml_string : MuJoCo XML model as a string
        timestep   : Optional override for the simulation timestep
        
    Returns:
        Dictionary containing the MuJoCo model and data instances
    """
    model = mujoco.MjModel.from_xml_string(xml_string)
    
    if timestep is not None:
        model.opt.timestep = timestep
    
    return {
        "model" : model,
        "data"  : mujoco.MjData(model)
    }
