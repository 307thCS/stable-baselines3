import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

class FixDMCSObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.zero_keys = []
        for key in self.observation_space:
            if len(self.observation_space[key].shape) == 0:
                self.zero_keys.append(key)
                self.observation_space[key] = Box(shape=(1,), low=-np.inf, high=np.inf, dtype = np.float64)
    def observation(self, obs):
        for key in self.zero_keys:
            obs[key] = np.expand_dims(obs[key], axis=0)
        return obs