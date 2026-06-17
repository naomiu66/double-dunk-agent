import gymnasium as gym
import ale_py
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo, AtariPreprocessing

import numpy as np
import os

from collections import deque

import torch
import torch.nn as nn

frameskips = 3
model_path = f"{os.getcwd()}/models/ppo/pacman-agent.pth"

class PacmanAgent(nn.Module):
    def __init__(self, n_hid, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(frameskips, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),         
               
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, n_hid),
            nn.ReLU(),  
            nn.Linear(n_hid, n_out)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.head(x)

def choose_action(model, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(0, action_size)
       
    state = torch.tensor(state, dtype=torch.float32)
    state = state.unsqueeze(0)
    with torch.no_grad():
        out = model(state)
    return int(torch.argmax(out).item())

gym.register_envs(ale_py)

env = gym.make("ALE/Pacman-v5", render_mode="human", frameskip=1)
env = AtariPreprocessing(env, frame_skip=frameskips, scale_obs=True)

obs, info = env.reset()

done = False

action_size = env.action_space.n

agent = PacmanAgent(512, action_size)
agent.load_state_dict(torch.load(model_path))
stack = deque(maxlen=frameskips)

for _ in range(frameskips):
    stack.append(obs)

while not done:
    action = choose_action(agent, stack, 0)
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    stack.append(obs)
    
    done = terminated or truncated
    
    print(info)
    
