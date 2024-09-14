import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
references:
[1] actor-critic algorithm: https://hrl.boyuai.com/chapter/2/actor-critic%E7%AE%97%E6%B3%95
[2] environment: https://gymnasium.farama.org/environments/classic_control/cart_pole/
[3] classic paper: https://proceedings.neurips.cc/paper_files/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf
"""


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=2e-2)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.from_numpy(np.array(transition_dict['states'], dtype=np.float32)).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.from_numpy(np.array(transition_dict['rewards'], dtype=np.float32)).float().view(-1, 1).to(self.device)
        next_states = torch.from_numpy(np.array(transition_dict['next_states'], dtype=np.float32)).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).view(-1, 1).to(self.device)

        # one-step temporal difference
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)

        # actor loss
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())    # gradient of actor loss is the policy gradient

        # critic loss
        loss_fn = nn.MSELoss()
        critic_loss = loss_fn(self.critic(states), td_target.detach())

        # update network parameters
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()


def train():
    num_episodes = 1000   # update the network parameters using 'num_episodes' steps
    hidden_dim = 128
    gamma = 0.99
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = ActorCritic(state_dim, hidden_dim, action_dim, gamma, device)

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
        for _ in range(500):  # truncation: run 500 steps max so that we don't infinite loop while learning
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
        agent.update(transition_dict)  # update one step
        return_list.append(episode_return)

        # log results
        if (i_episode + 1) % 50 == 0:
            print(f'Episode {i_episode + 1}, return: {np.mean(return_list[-10:]):.2f}')

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Actor-Critic on {}'.format(env_name))
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
    test(agent)  # render


















