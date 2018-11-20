import time
import argparse
import gym
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from drawnow import drawnow
import matplotlib.pyplot as plt

last_score_plot = [0]
avg_score_plot = [0]


def draw_fig():
  plt.title('reward')
  plt.plot(last_score_plot, '-')
  plt.plot(avg_score_plot, 'r-')


parser = argparse.ArgumentParser(description='PyTorch DDPG solution of MountainCarContinuous-V0')

parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--tau', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--max_episode', type=int, default=200)
parser.add_argument('--max_explore_eps', type=int, default=200)

cfg = parser.parse_args()


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


# Simple Ornstein-Uhlenbeck Noise generator
# Reference: https://github.com/rllab/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
def OUNoise():
  theta = 0.15
  sigma = 0.3
  mu = 0
  state = 0
  while True:
    yield state
    state += theta * (mu - state) + sigma * np.random.randn()


class Actor(nn.Module):
  def __init__(self):
    super(Actor, self).__init__()
    self.fc_1 = nn.Linear(2, 64)
    self.fc_2 = nn.Linear(64, 32)
    self.fc_out = nn.Linear(32, 1, bias=False)
    init.xavier_normal_(self.fc_1.weight)
    init.xavier_normal_(self.fc_2.weight)
    init.xavier_normal_(self.fc_out.weight)

  def forward(self, x):
    out = F.elu(self.fc_1(x))
    out = F.elu(self.fc_2(out))
    out = F.tanh(self.fc_out(out))
    return out


class Critic(nn.Module):
  def __init__(self):
    super(Critic, self).__init__()
    self.fc_state = nn.Linear(2, 32)
    self.fc_action = nn.Linear(1, 32)
    self.fc = nn.Linear(64, 128)
    self.fc_value = nn.Linear(128, 1, bias=False)
    init.xavier_normal_(self.fc_state.weight)
    init.xavier_normal_(self.fc_action.weight)
    init.xavier_normal_(self.fc.weight)
    init.xavier_normal_(self.fc_value.weight)

  def forward(self, state, action):
    out_s = F.elu(self.fc_state(state))
    out_a = F.elu(self.fc_action(action))
    out = torch.cat([out_s, out_a], dim=1)
    out = F.elu(self.fc(out))
    out = self.fc_value(out)
    return out


def get_action(_actor, state):
  if not isinstance(state, torch.Tensor):
    state = torch.from_numpy(state).float().cuda()
  action = _actor(state)
  action = torch.clamp(action, float(env.action_space.low[0]), float(env.action_space.high[0]))
  return action


def get_q_value(_critic, state, action):
  if not isinstance(state, torch.Tensor):
    state = torch.from_numpy(state).float().cuda()
  if not isinstance(action, torch.Tensor):
    action = torch.from_numpy(action).float().cuda()
  q_value = _critic(state, action)
  return q_value


def update_actor(state):
  action = actor(state)
  action = torch.clamp(action, float(env.action_space.low[0]), float(env.action_space.high[0]))
  # using chain rule to calculate the gradients of actor
  q_value = -torch.mean(critic(state, action))
  actor_optimizer.zero_grad()
  q_value.backward()
  actor_optimizer.step()
  return


def update_critic(state, action, target):
  q_value = critic(state, action)
  loss = F.mse_loss(q_value, target)
  critic_optimizer.zero_grad()
  loss.backward()
  critic_optimizer.step()
  return


def soft_update(target, source, tau):
  for target_param, param in zip(target.parameters(), source.parameters()):
    target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


env = gym.make('MountainCarContinuous-v0')

actor = Actor().cuda()
actor_target = Actor().cuda()
critic = Critic().cuda()
critic_target = Critic().cuda()

actor_optimizer = optim.Adam(actor.parameters(), lr=cfg.lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=cfg.lr)


def main():
  # env = wrappers.Monitor(env,'./tmp/',force=True)
  state = env.reset()
  noise = OUNoise()

  iteration_now = 0
  iteration = 0
  episode = 0
  episode_score = 0
  episode_steps = 0
  memory_warmup = cfg.batch_size * 3

  memory = Memory(memory_size=10000)
  start_time = time.perf_counter()
  while episode < cfg.max_episode:
    print('\riter {}, ep {}'.format(iteration_now, episode), end='')
    action = get_action(actor, state).item()

    # blend determinstic action with random action during exploration
    if episode < cfg.max_explore_eps:
      p = episode / cfg.max_explore_eps
      action = action * p + (1 - p) * next(noise)

    next_state, reward, done, _ = env.step([action])
    memory.append([state, action, reward, next_state, done])

    if iteration >= memory_warmup:
      memory_batch = memory.sample_batch(cfg.batch_size)

      state_batch, \
      action_batch, \
      reward_batch, \
      next_state_batch, \
      done_batch = map(lambda x: torch.tensor(x).float().cuda(), zip(*memory_batch))

      action_next = get_action(actor_target, next_state_batch)

      # using discounted reward as target q-value to update critic
      Q_next = get_q_value(critic_target, next_state_batch, action_next).detach()
      Q_target_batch = reward_batch[:, None] + cfg.gamma * (1 - done_batch[:, None]) * Q_next
      update_critic(state_batch, action_batch[:, None], Q_target_batch)

      # the action corresponds to the state_batch now is nolonger the action stored in buffer,
      # so we need to use actor to compute the action first, then use the critic to compute the q-value
      update_actor(state_batch)

      # soft update
      soft_update(actor_target, actor, cfg.tau)
      soft_update(critic_target, critic, cfg.tau)

    episode_score += reward
    episode_steps += 1
    iteration_now += 1
    iteration += 1
    if done:
      print(', score {:8f}, steps {}, ({:2f} sec/eps)'.
            format(episode_score, episode_steps, time.perf_counter() - start_time))
      avg_score_plot.append(avg_score_plot[-1] * 0.99 + episode_score * 0.01)
      last_score_plot.append(episode_score)
      drawnow(draw_fig)

      start_time = time.perf_counter()
      episode += 1
      episode_score = 0
      episode_steps = 0
      iteration_now = 0

      state = env.reset()
      noise = OUNoise()

    else:
      state = next_state
  env.close()


if __name__ == '__main__':
  main()
  plt.pause(0)
