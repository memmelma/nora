import gym
import numpy as np

from gym import spaces


class TimeLimit(gym.Wrapper):
    # https://github.com/openai/gym/blob/0.23.0/gym/wrappers/time_limit.py
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class SparseReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        reward = float(info["success"])
        return obs, reward, done, info


class ActionRepeat(gym.Wrapper):
    def __init__(self, env, action_repeat):
        super().__init__(env)
        self._action_repeat = action_repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._action_repeat):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class NormalizeAction(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Only normalize bounded action dimensions
        space = env.action_space
        bounded = np.isfinite(space.low) & np.isfinite(space.high)
        self._action_space = gym.spaces.Box(
            low=np.where(bounded, -1, space.low),
            high=np.where(bounded, 1, space.high),
            dtype=np.float32,
        )
        self._low = np.where(bounded, space.low, -1)
        self._high = np.where(bounded, space.high, 1)

    def step(self, action):
        orig_action = (action + 1) / 2 * (self._high - self._low) + self._low
        return self.env.step(orig_action)


class MetaWorldWrapper(gym.Wrapper):
    def __init__(self, env, pixel_obs=False):
        # Clear settings to use Metaworld environments as standalone tasks
        env.unwrapped._set_task_called = True
        env.unwrapped.random_init = False
        env.unwrapped.max_path_length = np.inf

        super().__init__(env)
        self._pixel_obs = pixel_obs
        self._camera = "corner3"
        self._resolution = 64

        if pixel_obs:
            img_shape = (self._resolution, self._resolution, 3)
            self._observation_space = spaces.Box(
                low=0, high=255, shape=img_shape, dtype=np.uint8
            )

    def reset(self, **kwargs):
        state_obs = self.env.reset(**kwargs)
        if self._pixel_obs:
            obs = self.render("rgb_array")
        else:
            obs = state_obs
        return obs

    def step(self, action):
        state_obs, reward, done, info = self.env.step(action)
        if self._pixel_obs:
            obs = self.render("rgb_array")
            info["state_obs"] = state_obs
        else:
            obs = state_obs
        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            image = self.env.sim.render(
                camera_name=self._camera,
                width=self._resolution,
                height=self._resolution,
                depth=False,
            )
            return image
        else:
            return self.env.render(mode)


class TabletopStateObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env.reset()
        state_obs = self.env._get_state()
        self._observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=state_obs.shape,
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        return self.env._get_state()

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        state_obs = info["state_obs"]
        return state_obs, reward, done, info
