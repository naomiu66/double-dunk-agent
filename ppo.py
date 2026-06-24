import cv2
import gymnasium as gym
import ale_py
from gymnasium.wrappers import RecordVideo, AtariPreprocessing, FrameStackObservation
from gymnasium import spaces

import numpy as np
import os

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

frameskips = 4
stack_size = 8
num_recordings = 10
model_path = f"{os.getcwd()}/models/ppo-fine-tuned/pacman-agent-ft-v6.pth"
video_dir = f"{os.getcwd()}/data/ppo/eval"

class ResizeRender(gym.Wrapper):
    def render(self):
        frame = self.env.render()

        return cv2.resize(
            frame,
            (640, 840),
            interpolation=cv2.INTER_NEAREST
        )

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PacmanAgent(nn.Module):
    def __init__(self, n_hid, n_out):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(stack_size, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),         
               
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, n_hid)),
            nn.ReLU(),  
        )
        
        self.actor = layer_init(nn.Linear(n_hid, n_out), std=0.01)
        self.critic = layer_init(nn.Linear(n_hid, 1), std=1)                

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action(self, x):
        hidden = self.network(x)
        logits = self.actor(hidden)
        
        action = torch.argmax(logits, dim=1)
        
        return action

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

gym.register_envs(ale_py)

env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array", frameskip=1)
env = AtariPreprocessing(env, frame_skip=frameskips, scale_obs=True)
env = FrameStackObservation(env, stack_size=stack_size)
env = ResizeRender(env)
env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: True)

action_size = np.int64(5)

agent = PacmanAgent(512, action_size)
agent.load_state_dict(torch.load(model_path))
agent.eval()

for recording in range(num_recordings):
    total_reward = 0

    obs, _ = env.reset()
    
    done = False
    
    while not done:
            
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            action = agent.get_action(obs_tensor)
        
        obs, reward, terminated, truncated, info = env.step(action)
            
        total_reward += reward
        
        done = terminated or truncated    
    print(total_reward)
env.close()