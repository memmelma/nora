from gym.envs.registration import register

## off-policy variBAD benchmark

register(
    "PointRobot-v0",
    entry_point="envs.meta.toy_navigation.point_robot:PointEnv",
    kwargs={"max_episode_steps": 60, "n_tasks": 2},
)

register(
    "PointRobotSparse-v0",
    entry_point="envs.meta.toy_navigation.point_robot:SparsePointEnv",
    kwargs={"max_episode_steps": 60, "n_tasks": 2, "goal_radius": 0.2},
)

register(
    "Wind-v0",
    entry_point="envs.meta.toy_navigation.wind:WindEnv",
)

register(
    "HalfCheetahVel-v0",
    entry_point="envs.meta.wrappers:mujoco_wrapper",
    kwargs={
        "entry_point": "envs.meta.mujoco.half_cheetah_vel:HalfCheetahVelEnv",
        "max_episode_steps": 200,
    },
    max_episode_steps=200,
)

## on-policy variBAD benchmark

register(
    "AntDir-v0",
    entry_point="envs.meta.wrappers:mujoco_wrapper",
    kwargs={
        "entry_point": "envs.meta.mujoco.ant_dir:AntDirEnv",
        "max_episode_steps": 200,
        "forward_backward": True,
        "n_tasks": None,
    },
    max_episode_steps=200,
)

register(
    "CheetahDir-v0",
    entry_point="envs.meta.wrappers:mujoco_wrapper",
    kwargs={
        "entry_point": "envs.meta.mujoco.half_cheetah_dir:HalfCheetahDirEnv",
        "max_episode_steps": 200,
        "n_tasks": None,
    },
    max_episode_steps=200,
)

register(
    "HumanoidDir-v0",
    entry_point="envs.meta.wrappers:mujoco_wrapper",
    kwargs={
        "entry_point": "envs.meta.mujoco.humanoid_dir:HumanoidDirEnv",
        "max_episode_steps": 200,
        "n_tasks": None,
    },
    max_episode_steps=200,
)

## Franka push + infer dynamics

register(
    "RobotPushEnvRGB-v0",
    entry_point="envs.meta.tabletop.physics_env:PushEnv",
    kwargs={
        "n_tasks": 2,
        "max_episode_steps": 200,
        "assets_root": "envs/meta/tabletop/assets",
        "block_colors": ["red", "red", "red"],
        "goal_colors": ["green"],
        "vis_obs": True
    },
    max_episode_steps=200,
)

register(
    "RobotPushEnvState-v0",
    entry_point="envs.meta.tabletop.physics_env:PushEnv",
    kwargs={
        "n_tasks": 2,
        "max_episode_steps": 200,
        "assets_root": "envs/meta/tabletop/assets",
        "block_colors": ["yellow", "orange", "red"],
        "goal_colors": ["green"],
        "vis_obs": False
    },
    max_episode_steps=200,
)

register(
    "RobotPushEnvState-v1",
    entry_point="envs.meta.tabletop.physics_env:PushEnv",
    kwargs={
        "n_tasks": 2,
        "max_episode_steps": 500,
        "assets_root": "envs/meta/tabletop/assets",
        "block_colors": ["yellow", "orange", "red"],
        "goal_colors": ["green"],
        "vis_obs": False
    },
    max_episode_steps=500,
)

register(
    "RobotPushEnvState-v2",
    entry_point="envs.meta.tabletop.physics_env:PushEnv",
    kwargs={
        "n_tasks": 2,
        "max_episode_steps": 500,
        "assets_root": "envs/meta/tabletop/assets",
        "block_colors": ["yellow", "red"],
        "goal_colors": ["green"],
        "vis_obs": False
    },
    max_episode_steps=500,
)