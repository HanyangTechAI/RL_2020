import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np

import gym

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 2)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))

        return F.softmax(self.fc2(x), dim=0)

class Agent:
    def __init__(self):
        self.learning_rate = 1e-3
        self.discount_factor = 0.99

        self.net = PolicyNet()
        self.opt = optim.Adam(self.net.parameters(), lr=self.learning_rate)

        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        policy = self.net(state)

        m = Categorical(policy)

        action = m.sample()
        self.log_probs.append(m.log_prob(action))

        return action.item()

    def train(self):
        R = 0
        returns = []

        for r in self.rewards[::-1]:
            R = r + self.discount_factor * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        policy_loss = []
        for log_prob, r in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * r)

        self.opt.zero_grad()

        loss = sum(policy_loss)
        loss.backward()

        self.opt.step()

        self.log_probs.clear()
        self.rewards.clear()

env = gym.make('CartPole-v1')

agent = Agent()
agent.net.load_state_dict(torch.load('checkpoint.bin'))

state = env.reset()
done = False
total_reward = 0

while not done:
    state = torch.FloatTensor(state)
    action = agent.select_action(state)

    next_state, reward, done, _ = env.step(action)
    total_reward += reward

    state = next_state
    env.render()

print("Total reward:", total_reward)
print("Press enter to exit", end='')
input()

env.close()
