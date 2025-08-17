"""
Custom experience collector for murmuration training.

This module provides a specialized collector that bypasses TorchRL's
SyncDataCollector to avoid signature mismatches and observation caching bugs.
The collector directly manages environment-policy interaction, ensuring proper
temporal progression and data diversity for imitation learning.
"""
from __future__ import annotations
from tensordict import TensorDict, stack
from torch.nn   import Module
from typing     import TYPE_CHECKING, Iterator

import torch as th

if TYPE_CHECKING:
    from thermur.imitation.controller import MurmurationController
    from thermur.imitation.simulation import SimulationEnv


class ExperienceCollector(Module):
    """
    Direct experience collector for murmuration imitation learning.
    
    This collector provides a clean interface between the simulation environment
    and expert policy, avoiding TorchRL's automatic policy wrapping and the
    observation caching bug present in SyncDataCollector with trust_policy=True.
    
    The collector manages trajectory rollouts, ensuring proper timestep
    progression and maintaining consistent data structure across all collected
    experiences. It yields batches of (observation, action, next_observation)
    tuples suitable for supervised learning and replay buffer storage.
    
    Key features:
        - Direct environment stepping without TorchRL wrappers
        - Proper extraction of next observations from environment outputs
        - Consistent experience structure with core observation keys
        - Automatic trajectory reset on termination or max length
        - Efficient batch stacking using TensorDict operations
    """
    
    def __init__(
        self,
        env:                 SimulationEnv,
        expert:              MurmurationController,
        frames_per_batch:    int,
        max_frames_per_traj: int,
        total_frames:        int
    ):
        """
        Initialize the experience collector with environment and expert.

    Attributes:
        env                 : The simulation environment instance
        expert              : The expert policy module
        frames_collected    : Running count of collected experience frames
        frames_per_batch    : Batch size for experience collection
        max_frames_per_traj : Trajectory length limit
        total_frames        : Target number of frames to collect
        """
        super().__init__()
        self.env                 = env
        self.expert              = expert
        self.frames_collected    = 0
        self.frames_per_batch    = frames_per_batch
        self.max_frames_per_traj = max_frames_per_traj
        self.total_frames        = total_frames
    
    def __iter__(self) -> Iterator[TensorDict]:
        """
        Iterate over collected experience batches.
        
        Yields batches of experiences until total_frames is reached. Each batch
        contains frames_per_batch transitions with consistent structure suitable
        for training. Trajectories are automatically reset when terminated or
        when max_frames_per_traj is reached.
        
        The method extracts observations from TorchRL's wrapped responses,
        maintains temporal consistency, and ensures all experiences have
        identical key structure for proper batch stacking.
        
        Yields:
            TensorDict containing stacked experiences with shape
            [frames_per_batch, ...] including current observations,
            actions taken, and next observations.
        """
        core_keys = [
            "action", "battery", "done", "edge_index", "gradient",
            "position", "reward", "temperature", "timestep",
            "trajectory_id", "velocity", "wind"
        ]
        
        while self.frames_collected < self.total_frames:
            batch_data = []
            obs = self.env.reset()
            traj_length = 0
            
            for _ in range(self.frames_per_batch):
                with th.no_grad():
                    action = self.expert(obs.clone()).get("action")
                
                step_result = self.env.step(TensorDict({"action": action}, batch_size=[]))
                next_obs    = step_result.get("next", step_result)
                
                obs["action"] = action
                
                experience = TensorDict({
                    **{
                        key: (val.clone() if th.is_tensor(val) else val)
                        for key in core_keys
                        if (val := obs.get(key)) is not None
                           or (val := self._default_value(key)) is not None
                    },
                    "next": TensorDict({
                        key: (val.clone() if th.is_tensor(val) else val)
                        for key in core_keys
                        if (val := next_obs.get(key)) is not None
                           or (val := self._default_value(key)) is not None
                    }, batch_size=[])
                }, batch_size=[])
                
                batch_data.append(experience)
                traj_length += 1
                
                if next_obs.get("done", th.tensor([False]))[0].item() or \
                   (self.max_frames_per_traj > 0 and traj_length >= self.max_frames_per_traj):
                    obs = self.env.reset()
                    traj_length = 0
                else:
                    obs = next_obs
            
            self.frames_collected += self.frames_per_batch
            yield stack(batch_data, dim=0) if batch_data else None
    
    def _default_value(self, key: str) -> th.Tensor | None:
        """
        Provide default values for missing observation keys.
        
        Ensures consistent data structure across all experiences by providing
        sensible defaults for optional keys that may not be present in every
        observation.
        
        Args:
            key: The observation key to provide a default for
        
        Returns:
            Default tensor value for the key, or None if no default needed
        """
        return {
            "done"   : th.tensor([False]),
            "reward" : th.zeros(self.env.flock.agent_count)
        }.get(key)
    
    def shutdown(self):
        """
        Clean shutdown for compatibility with Lightning's teardown hooks.
        """
        pass