import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

import sklearn
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler

from drawnow import drawnow, figure
import matplotlib.pyplot as plt

last_score_plot = [0]
avg_score_plot = [0]


def draw_fig():
  plt.title('reward')
  plt.plot(last_score_plot, '-')
  plt.plot(avg_score_plot, 'r-')


parser = argparse.ArgumentParser(description='PyTorch A2C solution of MountainCarContinuous-V0')
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--actor_lr', type=float, default=1e-4)
parser.add_argument('--critic_lr', type=float, default=5e-4)
parser.add_argument('--max_episode', type=int, default=100)

cfg = parser.parse_args()

env = gym.make('MountainCarContinuous-v0')

observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)
# env.seed(args.seed)
# torch.manual_seed(args.seed)
featurizer = sklearn.pipeline.FeatureUnion([
  ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
  ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
  ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
  ("rbf4", RBFSampler(gamma=0.5, n_components=100))
])
featurizer.fit(scaler.transform(observation_examples))


def process_state(state):
  scaled = scaler.transform([state])
  featurized = featurizer.transform(scaled)
  return featurized[0]


class Actor(nn.Module):
  def __init__(self):
    super(Actor, self).__init__()
    self.fc_mu = nn.Linear(400, 1)
    self.fc_sigma = nn.Linear(400, 1)
    init.xavier_normal_(self.fc_mu.weight)
    init.xavier_normal_(self.fc_sigma.weight)

  def forward(self, x):
    mu = self.fc_mu(x)
    sigma = F.softplus(self.fc_sigma(x)) + 1e-5
    return mu, sigma


class Critic(nn.Module):
  def __init__(self):
    super(Critic, self).__init__()
    self.fc_value = nn.Linear(400, 1)
    init.xavier_normal_(self.fc_value.weight)

  def forward(self, x):
    value = self.fc_value(x)
    return value


def get_action(state):
  state = torch.from_numpy(process_state(state)).float()
  action_mu, action_sigma = actor(state)
  action_dist = torch.distributions.normal.Normal(action_mu, action_sigma)
  action = action_dist.sample()
  action = torch.clamp(action, float(env.action_space.low[0]), float(env.action_space.high[0]))
  return action.item()


def get_state_value(state):
  state = torch.from_numpy(process_state(state)).float()
  state_value = critic(state)
  return state_value.item()


def update_actor(state, action, advantage):
  state = torch.from_numpy(process_state(state)).float()
  action_mu, action_sigma = actor(state)
  action_dist = torch.distributions.normal.Normal(action_mu, action_sigma)
  act_loss = -action_dist.log_prob(torch.tensor(action)) * advantage
  entropy = action_dist.entropy()
  loss = act_loss - 1e-4 * entropy
  actor_optimizer.zero_grad()
  loss.backward()
  actor_optimizer.step()
  return


def update_critic(state, target):
  state = torch.from_numpy(process_state(state)).float()
  state_value = critic(state)
  loss = F.mse_loss(state_value, torch.tensor(target))
  critic_optimizer.zero_grad()
  loss.backward()
  critic_optimizer.step()
  return


actor = Actor()
critic = Critic()
actor_optimizer = optim.Adam(actor.parameters(), lr=cfg.actor_lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=cfg.critic_lr)


def main():
  stats = []
  for i_episode in range(cfg.max_episode):
    state = env.reset()
    episode_score = 0
    for t in count():
      action = get_action(state)
      next_state, reward, done, _ = env.step([action])
      episode_score += reward

      # env.render()

      target = reward + cfg.gamma * get_state_value(next_state)
      td_error = target - get_state_value(state)

      update_actor(state, action, advantage=td_error)
      update_critic(state, target)

      if done:
        avg_score_plot.append(avg_score_plot[-1] * 0.99 + episode_score * 0.01)
        last_score_plot.append(episode_score)
        drawnow(draw_fig)
        break

      state = next_state

    stats.append(episode_score)
    if np.mean(stats[-100:]) > 90 and len(stats) >= 101:
      print(np.mean(stats[-100:]))
      print("Solved")
    print("Episode: {}, reward: {}.".format(i_episode, episode_score))
  return np.mean(stats[-100:])


if __name__ == '__main__':
  main()
  plt.pause(0)
