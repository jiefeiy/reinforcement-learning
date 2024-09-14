import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Deep deterministic policy gradient (DDPG) method for MDP with deterministic policy.
This algorithm can be viewed as an actor-critic method or a deep Q learning method.
It uses NN to approximate Q function and update Q function using current actor value.
It uses NN to approximate the policy and update the policy by optimizing Q function.
    - when there are a finite number of discrete actions, one can just compute the Q values for each action separately 
      and then compare them to get the state value function. (In classical AC, actor approximates distribution.)
    - when the action space is continuous, treating the optimization problem as a subroutine can be costly.
      In this case, DDPG can help. (In DDPG, actor approximates policy function or control function.)
      
Environment: 'Pendulum-v1'
    The pendulum starts in a random position and the goal is to apply torque on the free end to swing it into 
    an upright position, with its center of gravity right above the fixed point.
    
References:
[1] algorithm in chinese: https://hrl.boyuai.com/chapter/2/ddpg%E7%AE%97%E6%B3%95
[2] Open AI: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
[3] environment: https://gymnasium.farama.org/environments/classic_control/pendulum/
[4] paper 2014: https://proceedings.mlr.press/v32/silver14.pdf
[5] paper 2016: https://arxiv.org/pdf/1509.02971
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

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-3)

        self.action_dim = action_dim
        self.gamma = gamma
        self.sigma = sigma   # noise level
        self.tau = tau       # soft update

    def take_action(self, state):
        state = torch.from_numpy(np.array(state))
        action = self.actor(state).item()
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.from_numpy(np.array(transition_dict['states'], dtype=np.float32))
        actions = torch.from_numpy(np.array(transition_dict['actions'], dtype=np.float32)).view(-1, 1)
        next_states = torch.from_numpy(np.array(transition_dict['next_states'], dtype=np.float32))
        rewards = torch.from_numpy(np.array(transition_dict['rewards'], dtype=np.float32)).view(-1, 1)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).view(-1, 1)

        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_target = rewards + self.gamma * next_q_values * (1 - dones)

        # update critic
        critic_loss = F.mse_loss(self.critic(states, actions), q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target
        self.soft_update(self.critic, self.target_critic)
        self.soft_update(self.actor, self.target_actor)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)    # list-like container with fact appends and pops on either end

    def add(self, state, action, next_state, reward, done):
        self.buffer.append((state, action, next_state, reward, done))

    def sample(self, batch_size):  # sample a particular length of items chosen from the list, string, tuple, set ...
        transitions = random.sample(self.buffer, batch_size)
        states, actions, next_states, rewards, dones = zip(*transitions)
        return np.array(states), np.array(actions), np.array(next_states), np.array(rewards), dones

    def __len__(self):
        return len(self.buffer)


def train():
    num_episodes = 200
    hidden_dim = 64
    gamma = 0.98
    tau = 0.005

    buffer_size = 10000
    minimal_size = 1000   # the minimal length of replay_buffer
    batch_size = 64
    sigma = 0.01          # noise level

    env_name = 'Pendulum-v1'
    env = gym.make(env_name)

    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, tau, gamma)

    return_list = []
    for i_episode in range(num_episodes):
        episode_return = 0
        state, _ = env.reset()
        done = False

        for _ in range(200):     # truncation
            action = agent.take_action(state)
            next_state, reward, done, _, _ = env.step(action)
            replay_buffer.add(state, action, next_state, reward, done)

            state = next_state
            episode_return += reward
            if len(replay_buffer) > minimal_size:
                b_s, b_a, b_ns, b_r, b_d = replay_buffer.sample(batch_size)
                transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                agent.update(transition_dict)

            if done:
                break

        return_list.append(episode_return)

        # log results
        if (i_episode + 1) % 10 == 0:
            print(f'Episode {i_episode + 1}, return: {np.mean(return_list[-10:]):.2f}')

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DDPG on {}'.format(env_name))
    plt.show()
    return agent


def test(agent):
    env = gym.make('Pendulum-v1', render_mode="human")
    state, _ = env.reset()

    ep_total = 0
    for _ in range(200):
        action = agent.take_action(state)
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        ep_total += 1
        if done:
            break
    print(f"hold on for {ep_total} episodes")
    env.close()

    
if __name__ == '__main__':
    agent = train()
    test(agent)




