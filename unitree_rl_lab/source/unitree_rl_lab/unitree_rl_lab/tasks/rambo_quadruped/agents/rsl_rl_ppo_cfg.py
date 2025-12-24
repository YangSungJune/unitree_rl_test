# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RamboQuadrupedWalkPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Converted from your YAML to Isaac Lab RSL-RL configclass style."""

    # ===== seed =====
    seed = 42

    # ===== general =====
    experiment_name = "rambo_quadruped"
    run_name = "walk"
    max_iterations = 2000
    save_interval = 50
    empirical_normalization = True

    # 아래 값들은 보통 train.py의 CLI 인자/로직으로 처리된다.
    # - num_envs: env_cfg.scene.num_envs 또는 --num_envs
    # - video/video_length/video_interval: train.py argparse + RecordVideo wrapper
    # - logger/offline_mode: 프로젝트에 따라 runner/logger 구현이 달라서, 기본 cfg에 없을 수도 있음

    # ===== algorithm (PPO) =====
    # YAML: steps_per_env_iter = 24
    num_steps_per_env = 24

    # ===== network (ActorCritic) =====
    # YAML: log_std_init = 0.0  (즉 std = exp(0) = 1.0)
    # Isaac Lab cfg는 init_noise_std를 "표준편차"로 받는 경우가 일반적이라 1.0으로 변환
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        # losses and clipping
        value_loss_coef=1.0,
        use_clipped_value_loss=True,  # YAML: clip_value_target = True
        clip_param=0.2,               # YAML: clip_ratio = 0.2

        # entropy, optimization
        entropy_coef=0.001,           # YAML: entropy_coef
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        max_grad_norm=1.0,

        # discounting
        gamma=0.99,
        lam=0.95,

        # scheduler
        schedule="adaptive",
        desired_kl=0.02,
    )
