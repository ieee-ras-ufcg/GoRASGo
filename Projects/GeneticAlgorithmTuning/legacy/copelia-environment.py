import numpy as np
import gymnasium as gym
import random
from typing import Optional


class CopeliaEnvironment(gym.Env):
    def __init__(self, min_x=-1, min_y=-1, max_x=1, max_y=1):
        # super().__init__()
        self.minx = min_x
        self.miny = min_y
        self.maxx = max_x
        self.maxy = max_y
        self._agent_location = np.array([self.minx, self.miny, 0], dtype=np.float32)
        self._target_location = np.array([self.minx, self.miny], dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(
                    low=np.array([self.minx, self.miny, 0]),
                    high=np.array([self.maxx, self.maxy, 2 * np.pi]),
                    shape=(3,),
                    dtype=np.float32,
                ),
                "target": gym.spaces.Box(
                    low=np.array([self.minx, self.miny]),
                    high=np.array([self.maxx, self.maxy]),
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = gym.spaces.Box(shape=(2,), dtype=np.float32)

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location[:2] - self._target_location, ord=1
            )
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        self._target_location = self._agent_location

        self._target_location = self.np_random.random(2, size=2, dtype=np.float32)
        self._target_location[0] = self.minx + self._target_location[0] * (
            self.maxx - self.minx
        )
        self._target_location[1] = self.miny + self._target_location[1] * (
            self.maxy - self.miny
        )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    # TODO
    def step(self, action):
        # pseudocode
        copelia.step(action)
        self._agent_location = copelia.get_location()

        terminated = np.array_equal(self._agent_location[2:], self._target_location)
        truncated = False
        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info


gym.register(
    id="gymnasium_env/CopeliaEnvironment",
    entry_point=CopeliaEnvironment,
)
