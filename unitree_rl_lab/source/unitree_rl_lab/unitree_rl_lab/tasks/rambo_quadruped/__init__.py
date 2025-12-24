# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from .qp_env import QPEnv, QPEnvCfg


gym.register(
    id="Unitree-RAMBO-Go2-v0",
    entry_point=f"{__name__}.qp_env:QPEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QPEnvCfg,
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.rambo_quadruped.agents.rsl_rl_ppo_cfg:RamboQuadrupedWalkPPORunnerCfg",
    },
)
