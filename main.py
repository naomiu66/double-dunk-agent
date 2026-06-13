import gymnasium as gym
import ale_py
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

# import torch
# import torch.nn as nn

gym.register_envs(ale_py)

env = gym.make("ALE/DoubleDunk-v5", render_mode="human")

obs, info = env.reset()

done = False

while not done:
    action = env.action_space.sample()
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    done = terminated or truncated
    
