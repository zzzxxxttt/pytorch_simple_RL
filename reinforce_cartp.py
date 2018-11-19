import gym
import time
import argparse
import matplotlib.pyplot as plt
from drawnow import drawnow

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

last_score_plot = []
avg_score_plot = []


def draw_fig():
  plt.ylabel('reward')
  plt.xlabel('episode')
  plt.plot(last_score_plot, '-')
  plt.plot(avg_score_plot, 'r-')


parser = argparse.ArgumentParser(description='PyTorch REINFORCE solution of CartPole-V0')
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--max_episode', type=int, default=1000)
cfg = parser.parse_args()


class Policy(nn.Module):
  def __init__(self):
    super(Policy, self).__init__()
    self.fc1 = nn.Linear(4, 128)
    self.fc2 = nn.Linear(128, 2)

  def forward(self, x):
    x = F.elu(self.fc1(x))
    action_probs = F.softmax(self.fc2(x), dim=1)
    return action_probs


env = gym.make('CartPole-v0')
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)


def get_action(state):
  action_probs = policy(state)
  action_dist = torch.distributions.Categorical(action_probs)
  action = action_dist.sample()
  return action.item()


def update_policy(states, actions, returns):
  action_probs = policy(states)
  action_dist = torch.distributions.Categorical(action_probs)
  action_loss = -action_dist.log_prob(actions) * returns
  entropy = action_dist.entropy()
  loss = torch.mean(action_loss - 1e-4 * entropy)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  return


def main():
  state = env.reset()

  episode = 0
  episode_score = 0
  episode_steps = 0

  states = []
  actions = []
  rewards = []
  start_time = time.perf_counter()

  while episode < cfg.max_episode:
    action = get_action(torch.from_numpy(state).float()[None, :])

    next_state, reward, done, _ = env.step(action)
    states.append(state)
    actions.append(action)
    rewards.append(reward)

    episode_score += reward
    episode_steps += 1

    if done:
      # calculate discounted reward
      returns = [rewards[-1]]
      for r in rewards[-2::-1]:
        returns.append(r + cfg.gamma * returns[-1])

      state_batch = torch.tensor(states).float()
      action_batch = torch.tensor(actions).float()
      return_batch = torch.tensor(returns[::-1]).float()
      return_batch = (return_batch - return_batch.mean()) / return_batch.std()
      update_policy(state_batch, action_batch[:, None], return_batch)

      print('episode: %d score %.5f, steps %d, (%.2f sec/eps)' %
            (episode, episode_score, episode_steps, time.perf_counter() - start_time))
      last_score_plot.append(episode_score)
      if len(avg_score_plot) == 0:
        avg_score_plot.append(episode_score)
      else:
        avg_score_plot.append(avg_score_plot[-1] * 0.99 + episode_score * 0.01)
      drawnow(draw_fig)

      start_time = time.perf_counter()
      episode += 1
      episode_score = 0
      episode_steps = 0

      state = env.reset()
      states.clear()
      actions.clear()
      rewards.clear()
    else:
      state = next_state
  env.close()


if __name__ == '__main__':
  main()
  plt.pause(0)
