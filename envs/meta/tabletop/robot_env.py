import abc
import gym
import numpy as np
import os
import pkgutil
import pybullet as p

from gym import spaces

try:
    from .utils.cameras import DefaultCamera
except:
    from utils.cameras import DefaultCamera

PYBULLET_CONNECTION_MODE = os.environ["PYBULLET_CONNECTION_MODE"]
PYBULLET_RENDERER = os.environ["PYBULLET_RENDERER"]


class RobotEnv(gym.Env):
    def __init__(self, vis_obs=False, assets_root="assets", max_episode_steps=200):
        self.assets_root = assets_root
        
        self.vis_obs = vis_obs
        self.camera = DefaultCamera()
        self.render_size = (64, 64)

        self.action_space = spaces.Box(-1, 1, (5,), dtype=np.float32)
        self.reset_state = None

        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

        self.plane_pos = [0, 0, -0.001]
        self.workspace_pos = [0.5, 0, 0]
        # Start PyBullet
        if PYBULLET_CONNECTION_MODE == "gui":
            client = p.connect(p.SHARED_MEMORY)
            if client < 0:
                p.connect(p.GUI)
                # p.connect(p.GUI, options="--opengl2")


            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=self.camera.distance,
                cameraYaw=self.camera.yaw,
                cameraPitch=self.camera.pitch,
                cameraTargetPosition=self.camera.target,
            )
        elif PYBULLET_CONNECTION_MODE == "direct":
            p.connect(p.DIRECT)
            # Load EGL plugin for headless rendering
            self.egl_plugin = None
            if PYBULLET_RENDERER == "egl":
                egl = pkgutil.get_loader("eglRenderer")
                self.egl_plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        else:
            raise ValueError("Unsupported PyBullet connection mode")

    def reset(self):

        self._elapsed_steps = 0

        if self.reset_state is None:
            # Reset simulation
            p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
            p.setGravity(0, 0, -9.8)
            # Load scene
            p.loadURDF(
                os.path.join(self.assets_root, "plane/plane.urdf"), self.plane_pos
            )
            p.loadURDF(
                os.path.join(self.assets_root, "workspace/workspace.urdf"), self.workspace_pos
            )
            # Load robot
            self.robot = FrankaPanda(self.assets_root)
            # Load objects
            self._load_objects()
            # Save state for fast reset
            self.reset_state = p.saveState()
        else:
            p.restoreState(self.reset_state)
        
        self._set_object_color()

        if self.vis_obs:
            return self._get_obs()
        else:
            return self._get_state()

    def step(self, action):
        
        action = np.clip(action, -1, 1)
        self.robot.apply_action(action)

        if self.vis_obs:
            obs = self._get_obs()
        else:
            obs = self._get_state()

        reward, success = self._compute_reward()
        info = {"success": success, "state_obs": self._get_state()}

        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
        # if self._elapsed_steps >= self._max_episode_steps and auto_reset:
            # info["TimeLimit.truncated"] = not done
            done = True
        else:
            done = False
        return obs, reward, done, info

    def close(self):
        if PYBULLET_CONNECTION_MODE == "direct" and self.egl_plugin:
            p.unloadPlugin(self.egl_plugin)
        p.disconnect()

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            raise NotImplementedError("Unsupported render mode")
        _, _, image, _, _ = p.getCameraImage(
            width=self.render_size[1],
            height=self.render_size[0],
            viewMatrix=self.camera.view_matrix,
            projectionMatrix=self.camera.proj_matrix,
            flags=p.ER_NO_SEGMENTATION_MASK,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        image_size = self.render_size + (4,)
        image = np.array(image, dtype=np.uint8).reshape(image_size)
        image = image[:, :, :3]
        return image.transpose(2,0,1)

    def _get_obs(self):
        return self.render().reshape(-1)

    def _get_state(self):
        robot_state = self.robot.get_joint_pos()
        object_states = self._get_object_states()
        return np.concatenate((robot_state, object_states), 0)

    @abc.abstractmethod
    def _compute_reward(self):
        pass

    @abc.abstractmethod
    def _load_objects(self):
        pass

    @abc.abstractmethod
    def _set_object_color(self):
        pass

    @abc.abstractmethod
    def _get_object_states(self):
        pass


class DummyEnv(RobotEnv):
    def _load_objects(self):
        pass

    def _compute_reward(self):
        return 0, False


class FrankaPanda:
    def __init__(self, assets_root="assets"):
        self.assets_root = assets_root
        self.action_scale = np.array([0.01, 0.01, 0.01, 0.1, 0.01], dtype=np.float32)
        self.bounds = np.array([[0.25, -0.5, 0.008], [0.75, 0.5, 0.3]])
        self.sim_steps = 10

        # Load robot
        self.body = p.loadURDF(
            os.path.join(self.assets_root, "franka_panda/panda.urdf"),
            useFixedBase=True,
        )

        # Collect movable joints
        num_joints = p.getNumJoints(self.body)
        self.joints = []
        for joint in range(num_joints):
            if p.getJointInfo(self.body, joint)[2] != p.JOINT_FIXED:
                p.enableJointForceTorqueSensor(self.body, joint, 1)
                self.joints.append(joint)
        self.ee_angle = 6
        self.ee_grippers = [9, 10]
        self.ee_center = 11

        # Joint limits, ranges, and resting pose for null space
        self.ll = [-0.96, -1.83, -0.96, -3.14, -1.57, 0, -1.57, 0, 0]
        self.ul = [0.96, 1.83, 0.96, 0, 1.57, 3.8, 1.57, 0.04, 0.04]
        self.jr = [u - l for (u, l) in zip(self.ul, self.ll)]
        self.rp = [0, 0, 0, -0.75 * np.pi, 0, 0.75 * np.pi, 0, 0.04, 0.04]

        # Reset joint positions
        for j in range(len(self.joints)):
            p.resetJointState(self.body, self.joints[j], self.rp[j])

    def apply_action(self, action):
        # Scale actions
        action *= self.action_scale

        # Endeffector pose
        ee_pos, ee_orn = p.getLinkState(self.body, self.ee_center)[4:6]
        target_ee_pos = np.array(ee_pos) + action[:3]
        target_ee_pos = np.clip(target_ee_pos, self.bounds[0], self.bounds[1])
        # Keep endeffector facing downwards
        target_ee_orn = np.array(p.getEulerFromQuaternion(ee_orn))
        target_ee_orn[0] = -np.pi
        target_ee_orn = np.array(p.getQuaternionFromEuler(target_ee_orn))
        # Use regular IK because null space IK causes drifting
        target_joint_poses = p.calculateInverseKinematics(
            bodyUniqueId=self.body,
            endEffectorLinkIndex=self.ee_center,
            targetPosition=target_ee_pos,
            targetOrientation=target_ee_orn,
            maxNumIterations=100,
            residualThreshold=1e-4,
        )
        p.setJointMotorControlArray(
            bodyIndex=self.body,
            jointIndices=self.joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_joint_poses,
            targetVelocities=[0] * len(self.joints),
            forces=[200] * len(self.joints),
            positionGains=[0.4] * len(self.joints),
            velocityGains=[1] * len(self.joints),
        )

        # Endeffector angle
        ee_angle = p.getJointState(self.body, self.ee_angle)[0]
        target_ee_angle = np.clip(ee_angle + action[3], -1.57, 1.57)
        p.setJointMotorControl2(
            bodyIndex=self.body,
            jointIndex=self.ee_angle,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_ee_angle,
            targetVelocity=0,
            force=200,
            positionGain=0.8,
            velocityGain=1,
        )

        # Endeffector grippers
        # Make sure grippers are symmetric
        ee_gripper = p.getJointState(self.body, self.ee_grippers[0])[0]
        target_ee_gripper = np.clip(ee_gripper + action[4], 0, 0.04)
        p.setJointMotorControlArray(
            bodyIndex=self.body,
            jointIndices=self.ee_grippers,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[target_ee_gripper] * 2,
            targetVelocities=[0] * 2,
            forces=[40] * 2,
        )

        # Simulate for multiple steps
        for _ in range(self.sim_steps):
            p.stepSimulation()

    def get_joint_pos(self):
        joint_states = p.getJointStates(self.body, self.joints)
        joint_pos = [s[0] for s in joint_states]
        return np.array(joint_pos).astype(np.float32)
