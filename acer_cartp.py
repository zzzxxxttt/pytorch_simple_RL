import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque, namedtuple

import matplotlib.pyplot as plt
from drawnow import drawnow

last_score_plot = []
avg_score_plot = []


def draw_fig():
  plt.ylabel('reward')
  plt.xlabel('episode')
  plt.plot(last_score_plot, '-')
  plt.plot(avg_score_plot, 'r-')


# Parameters that work well for CartPole-v0
LEARNING_RATE = 1e-3
REPLAY_BUFFER_SIZE = 25
TRUNCATION_PARAMETER = 10
DISCOUNT_FACTOR = 0.99
REPLAY_RATIO = 4
MAX_EPISODES = 500
MAX_STEPS_BEFORE_UPDATE = 20
NUMBER_OF_AGENTS = 1
OFF_POLICY_MINIBATCH_SIZE = 16
TRUST_REGION_CONSTRAINT = 1.
TRUST_REGION_DECAY = 0.99
ENTROPY_REGULARIZATION = 1e-3
MAX_REPLAY_SIZE = 200
ACTOR_LOSS_WEIGHT = 0.1
render = True
verbose = True

Transition = namedtuple('Transition', ('states', 'actions', 'rewards', 'next_states',
                                       'done', 'exploration_statistics'))


class ReplayBuffer:
  def __init__(self):
    self.episodes = deque([[]], maxlen=REPLAY_BUFFER_SIZE)

  def add(self, transition):
    # add a new list if the last transition is the end of an episode
    if self.episodes[-1] and self.episodes[-1][-1].done[0, 0]:
      self.episodes.append([])
    self.episodes[-1].append(transition)

  def sample(self, batch_size, window_length=float('inf')):
    batched_trajectory = []
    trajectory_indices = np.random.choice(np.arange(len(self.episodes) - 1),
                                          size=min(batch_size, len(self.episodes) - 1),
                                          replace=False)
    trajectories = []
    for trajectory in [self.episodes[index] for index in trajectory_indices]:
      # sampled trajectory starts from a random position
      start = np.random.choice(np.arange(len(trajectory)), size=1)[0]
      trajectories.append(trajectory[start:start + window_length])
    smallest_trajectory_length = min([len(trajectory) for trajectory in trajectories]) if trajectories else 0
    # truncate trajectory to make sure all trajectories have the same length
    for index in range(len(trajectories)):
      trajectories[index] = trajectories[index][-smallest_trajectory_length:]
    # change trajectories from
    # [[t11, t12, t13],
    #  [t21, t22, t23]]
    # to
    # [[t11, t21],
    #  [t12, t22],
    #  [t13, t23]]
    for transitions in zip(*trajectories):
      batched_transition = Transition(*[torch.cat(data, dim=0) for data in zip(*transitions)])
      batched_trajectory.append(batched_transition)
    return batched_trajectory


class ActorCritic(nn.Module):
  # a network with two output branches
  def __init__(self):
    super().__init__()
    self.input_layer = torch.nn.Linear(4, 32)
    self.hidden_layer = torch.nn.Linear(32, 32)
    self.action_layer = torch.nn.Linear(32, 2)
    self.action_value_layer = torch.nn.Linear(32, 2)

  def forward(self, states):
    hidden = F.elu(self.input_layer(states))
    hidden = F.elu(self.hidden_layer(hidden))
    action_probabilities = F.softmax(self.action_layer(hidden))
    action_values = self.action_value_layer(hidden)
    return action_probabilities, action_values

  def copy_parameters_from(self, source, decay=0.):
    for parameter, source_parameter in zip(self.parameters(), source.parameters()):
      parameter.data.copy_(decay * parameter.data + (1 - decay) * source_parameter.data)

  def copy_gradients_from(self, source):
    for parameter, source_parameter in zip(self.parameters(), source.parameters()):
      parameter._grad = source_parameter.grad


env = gym.make('CartPole-v0')
buffer = ReplayBuffer()

actor_critic_ = ActorCritic()
average_actor_critic_ = ActorCritic()
average_actor_critic_.copy_parameters_from(actor_critic_)
optimizer = torch.optim.Adam(actor_critic_.parameters(), lr=LEARNING_RATE)


def learning_iteration(trajectory):
  actor_critic = ActorCritic()
  actor_critic.copy_parameters_from(actor_critic_)

  _, _, _, next_states, _, _ = trajectory[-1]
  action_probabilities, action_values = actor_critic(next_states)
  # state-value
  retrace_action_value = (action_probabilities * action_values).data.sum(-1).unsqueeze(-1)

  for states, actions, rewards, _, done, exploration_probabilities in reversed(trajectory):
    action_probabilities, action_values = actor_critic(states)
    average_action_probabilities, _ = average_actor_critic_(states)
    # expectation over a_t
    value = (action_probabilities * action_values).data.sum(-1).unsqueeze(-1) * (1. - done)
    action_indices = actions.long()

    importance_weights = action_probabilities.data / exploration_probabilities

    naive_advantage = action_values.gather(-1, action_indices).data - value

    retrace_action_value = rewards + DISCOUNT_FACTOR * retrace_action_value * (1. - done)
    retrace_advantage = retrace_action_value - value

    # Actor
    actor_loss = - ACTOR_LOSS_WEIGHT * \
                 importance_weights.gather(-1, action_indices.data).clamp(max=TRUNCATION_PARAMETER) * \
                 retrace_advantage * \
                 action_probabilities.gather(-1, action_indices).log()

    bias_correction = - ACTOR_LOSS_WEIGHT * \
                      (1 - TRUNCATION_PARAMETER / importance_weights).clamp(min=0.) * \
                      naive_advantage * \
                      action_probabilities.data * \
                      action_probabilities.log()

    actor_loss += bias_correction.sum(-1).unsqueeze(-1)
    actor_gradients = torch.autograd.grad(actor_loss.mean(), action_probabilities, retain_graph=True)
    actor_gradients = discrete_trust_region_update(actor_gradients, action_probabilities,
                                                   average_action_probabilities.data)
    action_probabilities.backward(actor_gradients, retain_graph=True)

    # Critic
    critic_loss = (action_values.gather(-1, action_indices) - retrace_action_value).pow(2)
    critic_loss.mean().backward(retain_graph=True)

    # Entropy
    entropy_loss = ENTROPY_REGULARIZATION * \
                   (action_probabilities * action_probabilities.log()).sum(-1)
    entropy_loss.mean().backward(retain_graph=True)

    retrace_action_value = importance_weights.gather(-1, action_indices.data).clamp(max=1.) \
                           * (retrace_action_value - action_values.gather(-1, action_indices).data) \
                           + value

  actor_critic_.copy_gradients_from(actor_critic)
  optimizer.step()
  average_actor_critic_.copy_parameters_from(actor_critic_, decay=TRUST_REGION_DECAY)


def discrete_trust_region_update(actor_gradients, action_probabilities, average_action_probabilities):
  negative_kullback_leibler = - ((average_action_probabilities.log() - action_probabilities.log())
                                 * average_action_probabilities).sum(-1)
  kullback_leibler_gradients = torch.autograd.grad(negative_kullback_leibler.mean(),
                                                   action_probabilities, retain_graph=True)
  updated_actor_gradients = []
  for actor_gradient, kullback_leibler_gradient in zip(actor_gradients, kullback_leibler_gradients):
    scale = actor_gradient.mul(kullback_leibler_gradient).sum(-1).unsqueeze(-1) - TRUST_REGION_CONSTRAINT
    scale = torch.div(scale, actor_gradient.mul(actor_gradient).sum(-1).unsqueeze(-1)).clamp(min=0.)
    updated_actor_gradients.append(actor_gradient - scale * kullback_leibler_gradient)
  return updated_actor_gradients


def main():
  env.reset()

  for episode in range(MAX_EPISODES):
    episode_rewards = 0.
    end_of_episode = False

    while not end_of_episode:
      state = torch.FloatTensor(env.env.state)
      trajectory = []
      for step in range(MAX_STEPS_BEFORE_UPDATE):
        action_probabilities, *_ = actor_critic_(state)
        action = action_probabilities.multinomial(1)

        exploration_statistics = action_probabilities.data.view(1, -1)
        next_state, reward, done, _ = env.step(action.numpy()[0])
        next_state = torch.from_numpy(next_state).float()
        if render:
          env.render()
        transition = Transition(states=state.view(1, -1),
                                actions=action.view(1, -1),
                                rewards=torch.FloatTensor([[reward]]),
                                next_states=next_state.view(1, -1),
                                done=torch.FloatTensor([[done]]),
                                exploration_statistics=exploration_statistics)
        buffer.add(transition)
        trajectory.append(transition)
        if done:
          env.reset()
          break
        else:
          state = next_state

      learning_iteration(trajectory)
      end_of_episode = trajectory[-1].done[0, 0]
      episode_rewards += sum([transition.rewards[0, 0] for transition in trajectory])
      for trajectory_count in range(np.random.poisson(REPLAY_RATIO)):
        if len(buffer.episodes) > 1:
          trajectory = buffer.sample(OFF_POLICY_MINIBATCH_SIZE, MAX_REPLAY_SIZE)
          if trajectory:
            learning_iteration(trajectory)

    if verbose:
      print("Episode #%d, episode rewards %d" % (episode, episode_rewards))
      last_score_plot.append(episode_rewards)
      if len(avg_score_plot) == 0:
        avg_score_plot.append(episode_rewards)
      else:
        avg_score_plot.append(avg_score_plot[-1] * 0.99 + episode_rewards * 0.01)
      drawnow(draw_fig)


if __name__ == '__main__':
  main()
  plt.pause(0)
