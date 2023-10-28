import gymnasium as gym
import numpy as np
from gymnasium import Wrapper, ObservationWrapper
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
        
class SkipFrameWrapper(Wrapper):
    def __init__(self, env, num_steps = 2):
        super().__init__(env)
        self.num_steps = num_steps
        self.total_steps = 0
    def step(self, action):
        total_reward = 0
        for i in range(self.num_steps):
            self.total_steps += 1
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated == True or truncated == True:
                return obs, total_reward, terminated, truncated, info
        return obs, total_reward, terminated, truncated, info