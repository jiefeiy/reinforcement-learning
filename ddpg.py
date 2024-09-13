from typing import Union, Callable, Tuple, Any, Optional, Dict

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T
from torch.utils.hooks import RemovableHandle

"""
Deep deterministic policy gradient (DDPG) method for MDP with deterministic policy.
This algorithm can be viewed as an actor-critic method or a deep Q learning method.
It uses NN to approximate Q function and update Q function using current actor value.
It uses NN to approximate the policy and update the policy by optimizing Q function.
    - when there are a finite number of discrete actions, one can just compute the Q values for each action separately 
      and then compare them to get the state value function. (In classical AC, actor approximates distribution.)
    - when the action space is continuous, treating the optimization problem as a subroutine can be costly.
      In this case, DDPG can help. (In DDPG, actor approximates policy function or control function.)
References:
[1] algorithm in chinese: https://hrl.boyuai.com/chapter/2/ddpg%E7%AE%97%E6%B3%95
[2] Open AI: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
[3] paper 2014: https://proceedings.mlr.press/v32/silver14.pdf
[4] paper 2016: https://arxiv.org/pdf/1509.02971
"""


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # fit the environment

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound   # output in [-action_bound, action_bound]


class QValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)   # input is (state, action)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DDPG:
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma, tau, gamma):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-2)

        self.action_dim = action_dim
        self.gamma = gamma
        self.sigma = sigma   # noise level
        self.tau = tau       # soft update

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float32)
        action = self.actor(state).item
        action += self.sigma * torch.randn(self.action_dim)
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    





