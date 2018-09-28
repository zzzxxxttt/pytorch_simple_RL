import argparse
import gym
import numpy as np
from collections import deque
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from drawnow import drawnow, figure
import matplotlib.pyplot as plt

last_score_plot = [0]
avg_score_plot = [0]


def draw_fig():
  plt.title('reward')
  plt.plot(last_score_plot, '-')
  plt.plot(avg_score_plot, 'r-')


parser = argparse.ArgumentParser(description='PyTorch A2C solution of CartPole-v0')
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--actor_lr', type=float, default=1e-4)
parser.add_argument('--critic_lr', type=float, default=5e-4)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--max_episode', type=int, default=500)

cfg = parser.parse_args()

env = gym.make('CartPole-v0')


class Memory(object):
  def __init__(self, memory_size=10000):
    self.memory = deque(maxlen=memory_size)
    self.memory_size = memory_size

  def __len__(self):
    return len(self.memory)

  def append(self, item):
    self.memory.append(item)

  def sample_batch(self, batch_size):
    idx = np.random.permutation(len(self.memory))[:batch_size]
    return [self.memory[i] for i in idx]


class Actor(nn.Module):
  def __init__(self):
    super(Actor, self).__init__()
    self.fc1 = nn.Linear(4, 64)
    self.fc2 = nn.Linear(64, 2)
    init.xavier_normal_(self.fc1.weight)
    init.xavier_normal_(self.fc2.weight)

  def forward(self, x):
    x = F.elu(self.fc1(x))
    x = F.softmax(self.fc2(x), dim=1)
    return x


class Critic(nn.Module):
  def __init__(self):
    super(Critic, self).__init__()
    self.fc1 = nn.Linear(4, 64)
    self.fc2 = nn.Linear(64, 1)
    init.xavier_normal_(self.fc1.weight)

  def forward(self, x):
    x = F.elu(self.fc1(x))
    value = self.fc2(x)
    return value.squeeze()


def get_action(state):
  action_probs = actor(state)
  action_dist = torch.distributions.Categorical(action_probs)
  action = action_dist.sample()
  return action


def get_state_value(state):
  state_value = critic(state)
  return state_value


def update_actor(states, actions, advantages):
  action_probs = actor(states)
  action_dist = torch.distributions.Categorical(action_probs)
  act_loss = -action_dist.log_prob(actions) * advantages
  entropy = action_dist.entropy()
  loss = torch.mean(act_loss - 1e-4 * entropy)
  actor_optimizer.zero_grad()
  loss.backward()
  actor_optimizer.step()
  return


def update_critic(states, targets):
  state_value = critic(states)
  loss = F.mse_loss(state_value, targets)
  critic_optimizer.zero_grad()
  loss.backward()
  critic_optimizer.step()
  return


actor = Actor()
critic = Critic()
actor_optimizer = optim.Adam(actor.parameters(), lr=cfg.actor_lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=cfg.critic_lr)

memory = Memory(10000)

for i in range(cfg.max_episode):
  episode_durations = 0
  state = env.reset()

  for t in count():
    action = get_action(torch.tensor(state).float()[None, :]).item()
    next_state, reward, done, _ = env.step(action)

    memory.append([state, action, next_state, reward, done])
    state = next_state

    if len(memory) > cfg.batch_size:
      states, actions, next_states, rewards, dones = \
        map(lambda x: torch.tensor(x).float(), zip(*memory.sample_batch(cfg.batch_size)))

      targets = rewards + cfg.gamma * get_state_value(next_states).detach() * (1 - dones)
      td_errors = targets - get_state_value(states).detach()

      update_actor(states=states, actions=actions, advantages=td_errors)
      update_critic(states, targets)

    if done:
      episode_durations = t + 1
      avg_score_plot.append(avg_score_plot[-1] * 0.99 + episode_durations * 0.01)
      last_score_plot.append(episode_durations)
      drawnow(draw_fig)
      break

print('Complete')
env.close()

plt.pause(0)
