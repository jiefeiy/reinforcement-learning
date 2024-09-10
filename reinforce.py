import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


# https://hrl.boyuai.com/chapter/2/%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E7%AE%97%E6%B3%95
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)


class Reinforce:
    def __init__(self, state_dim, hidden_dim, action_dim, gamma, device):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=3e-3)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dist):
        reward_list = transition_dist['rewards']
        state_list = transition_dist['states']
        action_list = transition_dist['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = torch.from_numpy(state_list[i]).unsqueeze(0).to(self.device)
            action = torch.tensor(action_list[i]).view(-1, 1).to(self.device)  # torch中的view相当于numpy中reshape
            log_prob = torch.log(self.policy_net(state).gather(1, action))  # 提取出每个action对应的prob，再取log
            G = self.gamma * G + reward   # \psi_t := \sum_{s=t}^T \gamma^{s-t} R_s
            loss = -log_prob * G
            loss.backward()
        self.optimizer.step()


def train():
    num_episodes = 1000  # 迭代1000次
    hidden_dim = 128
    gamma = 0.99
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    env.reset()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = Reinforce(state_dim, hidden_dim, action_dim, gamma, device)

    return_list = []
    for i_episode in range(num_episodes):
        episode_return = 0
        transition_dict = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': []
        }
        state, _ = env.reset()
        done = False
        for t in range(500):  # for each episode, only run 500 steps so that we don't infinite loop while learning
            action = agent.take_action(state)
            next_state, reward, done, _, _ = env.step(action)
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            state = next_state
            episode_return += reward
            if done:
                break

        return_list.append(episode_return)
        agent.update(transition_dict)

        # log results
        if (i_episode + 1) % 50 == 0:
            print(f'Episode {i_episode + 1}, return: {np.mean(return_list[-10:]):.2f}')

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('REINFORCE on {}'.format(env_name))
    plt.show()
    return agent


def test(agent):
    env = gym.make("CartPole-v1", render_mode="human")
    state, _ = env.reset()

    r = 0
    for _ in range(500):
        action = agent.take_action(state)
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        r += 1
        if done:
            break

    print(f"The test reward is {r}.")

    env.close()


if __name__ == '__main__':
    agent = train()
    test(agent)   # render


