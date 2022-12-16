import numpy as np
import os
import pybullet as p

try:
    from .robot_env import RobotEnv
    from .wrappers import TimeLimit
    from .utils.colors import COLORS
except:
    from robot_env import RobotEnv
    from wrappers import TimeLimit
    from utils.colors import COLORS

from gym import spaces

class PushEnv(RobotEnv):
    def __init__(
        self,
        n_tasks=2,
        max_episode_steps=200,
        block_colors = ["yellow", "orange", "red"],
        goal_colors = ["green"],
        vis_obs=False,
        assets_root="assets",
    ):
        super().__init__(vis_obs=vis_obs, assets_root=assets_root, max_episode_steps=max_episode_steps)

        self.n_tasks = n_tasks
        self.step_count = 0

        self.block_colors = block_colors
        self.goal_colors = goal_colors

        self.block_urdf = os.path.join(self.assets_root, "block/block.urdf")
        self.goal_urdf = os.path.join(self.assets_root, "cross/cross.urdf")

        self.x_low, self.x_high = 0.3, 0.7
        self.y_low, self.y_high = -0.4, 0.4
        self.z_low, self.z_high = 0.0, 0.02
        # self.mass_low, self.mass_high = 1., 10.
        self.mass_low, self.mass_high = np.array([[1.3, 0.7, 0.1]]), np.array([[1.5, 0.9, 0.3]])

        self.goal_pos = np.random.uniform(low=[self.x_low, self.y_low, self.z_low], high=[self.x_high, self.y_high, self.z_low], size=(n_tasks,len(self.goal_colors),3))
        
        self.goal_pos = np.array([[0.5, 0.0, 0.0]])
        self.goal_pos = np.repeat(self.goal_pos[np.newaxis,:,:], n_tasks, 0)

        # create coordinates for each block
        self.block_pos = np.random.uniform(low=[self.x_low, self.y_low, 0.02], high=[self.x_high, self.y_high, 0.02], size=(n_tasks,len(self.block_colors),3))
        d_goal = 0.1
        d_diag = np.sin(np.radians(45)) * d_goal
        self.block_pos = np.array([self.goal_pos[0][0] + [0.1, 0.0, 0.02], self.goal_pos[0][0] + [+d_diag, -d_diag, 0.02], self.goal_pos[0][0] + [+d_diag, +d_diag, 0.02]])
        self.block_pos = np.repeat(self.block_pos[np.newaxis,:,:], n_tasks, 0)
        self.block_pos = self.block_pos[:,:len(self.block_colors)]
        
        # self.masses = np.random.uniform(low=self.mass_low, high=self.mass_high, size=(n_tasks, len(self.block_colors)))
        self.masses = np.random.uniform(low=np.repeat(self.mass_low[np.newaxis,:,:len(self.block_colors)], n_tasks, 0), high=np.repeat(self.mass_high[np.newaxis,:,:len(self.block_colors)], n_tasks, 0))
        for idx in range(n_tasks):
            np.random.shuffle(self.masses[idx].transpose(1,0))
        
        if vis_obs:
            # (C, H, W)
            self.image_space = (3,) + self.render_size
            self.observation_space = spaces.Box(
                0, 255, [np.prod(self.image_space)], dtype=np.uint8
            )
        else:
            self.observation_space = spaces.Box(
                # joint_dim + block_pos_rot (- 1 if euler instead of quaternions) + goal_pos
                # w/o linear and angular velocity of objects
                # 0, 255, [9 + len(block_colors) * 7 + len(goal_colors) * 3], dtype=np.float32
                # w/ linear and angular velocity of objects
                0, 255, [9 + len(block_colors) * 13 + len(goal_colors) * 3], dtype=np.float32
            )

        self.reset_task(0)

    def reset_task(self, idx):
        """reset goal AND reset the agent"""

        # block initial values -> c.f. URDF
                # lateral_friction value="1.0"
                # rolling_friction value="0.0001"
                # inertia_scaling value="3.0"
                # mass value=".1"
            # mass, lateralFriction, _, _, _, _, rollingFriction, spinningFriction, _, _, bodytype, _ = p.getDynamicsInfo(block_id, -1)
        if idx is not None:
            self._block_pos = self.block_pos[idx]
            self._goal_pos = self.goal_pos[idx]
            self._masses = self.masses[idx][0]
        self.reset()

    def get_all_task_idx(self):
        return range(self.n_tasks)

    def _load_objects(self):

        self.blocks = []
        self.goals = []
        # dynamic objects, w/ goal pos
        for idx in range(len(self.block_colors)):
            # block
            block_id = p.loadURDF(self.block_urdf, self._block_pos[idx])
            # p.changeVisualShape(block_id, -1, rgbaColor=COLORS[self.block_colors[idx]] + [1])
            p.changeDynamics(block_id, -1, mass=self._masses[idx])
            self.blocks.append(block_id)

        for idx in range(len(self.goal_colors)):
            goal_id = p.loadURDF(self.goal_urdf, self._goal_pos[idx], useFixedBase=True)
            p.changeVisualShape(goal_id, -1, rgbaColor=COLORS[self.goal_colors[idx]] + [2])
            self.goals.append(goal_id)

    def _set_object_color(self):
        
        color_idx = np.argsort(self._masses)
        for idx, block_id in enumerate(self.blocks):
            p.changeVisualShape(block_id, -1, rgbaColor=COLORS[self.block_colors[color_idx[idx]]] + [1])

    def _compute_reward(self):
        
        self.highest_mass = np.argmax(self._masses)
        self.lowest_mass = np.argmin(self._masses)

        # endeffector center position
        ee_center_pos = np.array(
            p.getLinkState(self.robot.body, self.robot.ee_center)[0]
        )
        # endeffector gripper position
        ee_gripper_pos = np.array(
            p.getLinkState(self.robot.body, self.robot.ee_grippers[0])[0]
        )

        block_poss = np.array([p.getBasePositionAndOrientation(block_id)[0] for block_id in self.blocks])
        goal_poss = np.array([p.getBasePositionAndOrientation(goal_id)[0] for goal_id in self.goals])
        # obj_pos_0 = np.array(p.getBasePositionAndOrientation(self.blocks[0])[0])
        # obj_pos_1 = np.array(p.getBasePositionAndOrientation(self.blocks[1])[0])
        # goal_pos = np.array(p.getBasePositionAndOrientation(self.goals[0])[0])
        
        # incentivize reaching for both fixed and dynamic objects
        reach_dist = 0
        for block_pos in block_poss:
            reach_dist += np.linalg.norm(ee_center_pos - block_pos)
        # # incentivize reaching for both fixed and dynamic objects
        # reach_dist_0 = np.linalg.norm(ee_center_pos - obj_pos_0)
        # reach_dist_1 = np.linalg.norm(ee_center_pos - obj_pos_1)
        # reach_dist = np.min(reach_dist_0 + reach_dist_1)

        # only dynamic object can be pushed 
        push_dist = np.linalg.norm(block_poss[self.highest_mass] - goal_poss[0])

        # distance from endeffector center to gripper (small when gripper closed)
        grip_dist = np.linalg.norm(ee_gripper_pos - ee_center_pos)

        max_push_dist = np.linalg.norm(
            self._block_pos[self.highest_mass] - self._goal_pos[0]
        )

        reach_rew = -reach_dist
        if reach_dist < 0.03:
            # # incentive to close gripper when reach_dist is small
            reach_rew += np.exp(-(grip_dist**2) / 0.05)
            
            # only applicable for dyamic object
            push_rew = 1000 * (max_push_dist - push_dist) + 1000 * (
                np.exp(-(push_dist**2) / 0.01) + np.exp(-(push_dist**2) / 0.001)
            )
            push_rew = max(push_rew, 0)

        else:
            push_rew = 0

        success = float(push_dist < 0.05)

        reward = reach_rew + push_rew + success
        return reward, success

    def _get_object_states(self):
        object_states = []
        for block_id in self.blocks:
            pos, rot = p.getBasePositionAndOrientation(block_id)
            lin_velo, ang_velo = p.getBaseVelocity(block_id)
            # rot = np.array(p.getEulerFromQuaternion(rot)) / (2 * np.pi)

            # max = np.array([self.x_high, self.y_high, self.z_high])
            # min = np.array([self.x_low, self.y_low, self.z_low])
            # pos = np.array(pos)

            # pos_norm = (pos - min) / (max - min)
            object_states.extend([pos, rot, lin_velo, ang_velo])
        for goal_id in self.goals:
            pos = p.getBasePositionAndOrientation(goal_id)[0]

            # max = np.array([self.x_high, self.y_high, self.z_high])
            # min = np.array([self.x_low, self.y_low, self.z_low])
            # pos = np.array(pos)

            # pos_norm = (pos - min) / (max - min)
            object_states.append(pos)
        return np.concatenate(object_states, 0).astype(np.float32)