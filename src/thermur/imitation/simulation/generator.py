"""
MuJoCo XML generation utilities for the Thermur project.

This module provides a class-based builder for dynamically generating and
loading MuJoCo models with varying numbers of agents, enabling true multi-agent
physics simulation for flock environments.
"""
from functools import reduce
from itertools import chain
from operator  import methodcaller
from pathlib   import Path
from typing    import Any

import mujoco as mj


class XMLGenerator:
    """
    Dynamically generates MuJoCo XML models for multi-agent flock simulations.
    
    This generator encapsulates the logic for creating multi-agent MuJoCo
    simulations with configurable agent counts and spatial dimensions.
    It provides a clean interface for generating XML models and loading
    them into MuJoCo.
    
    Attributes:
        assets_dir     : Path to directory containing XML template files
        drone_template : Cached drone XML template string  
        flock_template : Cached flock environment XML template string
    """
    
    def __init__(self, assets_dir: Path):
        """
        Initialize the builder with XML templates from the assets directory.
        
        Args:
            assets_dir: Directory containing drone.xml and flock.xml templates
        """
        self.assets_dir = assets_dir
        self.drone_template, self.flock_template = self._load_templates()
    
    def _assemble_xml(
        self,
        actuator_defs   : list[str],
        drone_bodies    : list[str],
        simulation_step : float
    ) -> str:
        """
        Replaces placeholders in the flock template with generated drone
        bodies, actuators, and simulation parameters to create a complete
        MuJoCo model definition.
        
        Args:
            actuator_defs   : List of XML strings defining velocity actuators
            drone_bodies    : List of XML strings defining each drone body
            simulation_step : Physics integration timestep in seconds
            
        Returns:
            Complete MuJoCo XML model string ready for loading
        """
        replacements = [
            ("$TIMESTEP$", str(simulation_step)),
            ("<!-- DRONE_BODIES -->", "\n".join(drone_bodies)),
            ("<!-- ACTUATORS -->", "".join(actuator_defs))
        ]
        
        return reduce(
            lambda xml, r: xml.replace(*r), 
            replacements, 
            self.flock_template
        )
    
    def _generate_actuators(self, agent_idx: int, spatial_dims: int) -> list[str]:
        """
        Generate velocity actuator definitions for one agent's control.
        
        Creates independent velocity controllers for each spatial dimension,
        allowing full control over the agent's motion. Actuators are limited
        to [-1, 1] control range for normalized inputs.
        
        Args:
            agent_idx    : Zero-based index of the agent
            spatial_dims : Number of spatial dimensions (2 or 3)
            
        Returns:
            List of actuator XML strings for this agent
        """
        template = """
        <velocity
            ctrllimited = "true"
            ctrlrange   = "-1 1"
            gear        = "1"
            joint       = "drone_{agent}_joint_{dim}"
            name        = "drone_{agent}_vel_{axis}"
        />"""
        
        return [
            template.format(agent=agent_idx, dim=j, axis=axis)
            for j, axis in enumerate("xyz"[:spatial_dims])
        ]
    
    def _generate_drone_body(self, agent_idx: int, spatial_dims: int) -> str:
        """
        Generate the complete XML definition for a single drone body.
        
        Combines the drone template with agent-specific parameters including
        unique ID, initial position, and joint definitions to create a
        complete body definition.
        
        Args:
            agent_idx    : Zero-based index of the agent
            spatial_dims : Number of spatial dimensions (2 or 3)
            
        Returns:
            Complete XML string defining the drone body
        """
        replacements = [
            ("$AGENT_ID$", str(agent_idx)),
            ("$POSITION$", self._get_initial_position(agent_idx, spatial_dims)),
            ("<!-- JOINTS_XML -->", self._generate_joints(agent_idx, spatial_dims))
        ]
        
        return reduce(
            lambda body, r: body.replace(*r), replacements, self.drone_template
        )
    
    def _generate_joints(self, agent_idx: int, spatial_dims: int) -> str:
        """
        Generate slide joint definitions for one agent's degrees of freedom.
        
        Creates constrained slide joints for movement in each spatial
        dimension. Joints are limited to ±10 units of motion and named
        uniquely for each agent to enable individual control.
        
        Args:
            agent_idx    : Zero-based index of the agent
            spatial_dims : Number of spatial dimensions (2 or 3)
            
        Returns:
            XML string containing all joint definitions for this agent
        """
        axes     = ["1 0 0", "0 1 0", "0 0 1"][:spatial_dims]
        template = """
            <joint
                axis    = "{axis}"
                limited = "true"
                name    = "drone_{agent}_joint_{dim}"
                pos     = "0 0 0"
                range   = "-10 10"
                type    = "slide"
            />"""
        
        return "".join(
            template.format(agent=agent_idx, dim=j, axis=axis)
            for j, axis in enumerate(axes)
        )
    
    def _get_initial_position(self, agent_idx: int, spatial_dims: int) -> str:
        """
        Calculate the initial position for an agent to prevent collisions.
        
        Distributes agents along the Y-axis with sufficient spacing to
        prevent initial overlaps. The spacing is designed to work with
        the default drone geometry size.
        
        Args:
            agent_idx    : Zero-based index of the agent
            spatial_dims : Number of spatial dimensions (2 or 3)
            
        Returns:
            Position string formatted for MuJoCo XML (space-separated values)
        """
        offset = 0.3 * agent_idx
        return f"0 {offset} 0" if spatial_dims == 3 else f"0 {offset}"
    
    def _load_templates(self) -> tuple[str, str]:
        """
        Load XML templates from the assets directory.
        
        Reads drone.xml and flock.xml templates from disk and returns them
        as a tuple for unpacking during initialization.
        """
        read = methodcaller("read_text")
        return (
            read(self.assets_dir / "drone.xml"),
            read(self.assets_dir / "flock.xml")
        )

    def generate_xml(
        self,
        shape           : tuple[int, int],
        simulation_step : float
    ) -> str:
        """
        Generate a complete MuJoCo XML model with N distinct drone bodies.
        
        Creates a multi-agent simulation environment by iterating through
        each agent to generate bodies and actuators, then assembling all
        components into a valid MuJoCo model.
        
        Args:
            shape           : Tuple of (agent_count, spatial_dims) from FlockModel
            simulation_step : Physics simulation timestep in seconds
        
        Returns:
            Complete MuJoCo XML model as a string
        """
        n, dims = shape
        
        components = [
            (self._generate_drone_body(i, dims), self._generate_actuators(i, dims))
            for i in range(n)
        ]
        
        bodies, actuators = zip(*components) if components else ([], [])
        
        return self._assemble_xml(
            list(chain.from_iterable(actuators)),
            list(bodies),
            simulation_step
        )
    
    def load_model(self, xml_string: str) -> dict[str, Any]:
        """
        Load a MuJoCo model from an XML string.
        
        Creates initialized MuJoCo model and data objects from the provided
        XML definition. The data object is automatically sized to match the
        model and initialized with default values.
        
        Args:
            xml_string: Complete MuJoCo XML model definition
        
        Returns:
            Dictionary with 'model' and 'data' keys containing MuJoCo objects
        """
        if not (model := mj.MjModel.from_xml_string(xml_string)):
            raise ValueError("Failed to load MuJoCo model from XML")
        
        return {
            "data"  : mj.MjData(model), 
            "model" : model
        }