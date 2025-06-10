"""
Typed configuration objects stored via hydra-zen.

Only a single `TrainCfg` is registered for now, which can be overriden
via CLI:

    python -m thermur.train train.steps=5000 env.agent_count=40
"""

from hydra_zen         import make_config, store
from pydantic_settings import BaseSettings


class EnvCfg(BaseSettings, extra="forbid"):
    """
    Environment parameters.
    """
    name        : str   = "GaussianPlumeEnv"
    agent_count : int   = 20
    max_temp_f  : float = 480.0


TrainCfg = make_config(
    seed   = 0,
    env    = EnvCfg(),
    steps  = 1_000,
    lr     = 3e-4,
    device = "cpu",
)

store(
    group = "train", 
    name  = "base", 
    node  = TrainCfg
)