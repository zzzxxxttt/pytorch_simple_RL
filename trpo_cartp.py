import argparse
import gym
import time
import numpy as np
import scipy.optimize as opt

import torch
import torch.nn as nn
import torch.nn.functional as F

from drawnow import drawnow
import matplotlib.pyplot as plt

last_score_plot = [-100]
avg_score_plot = [-100]


def draw_fig():
  plt.title('reward')
  plt.plot(last_score_plot, '-')
  plt.plot(avg_score_plot, 'r-')


parser = argparse.ArgumentParser(description='PyTorch TRPO solution of MountainCarContinuous-v0')
parser.add_argument('--gamma', type=float, default=0.995)
parser.add_argument('--lambda', type=float, default=0.97)
parser.add_argument('--critic_wd', type=float, default=1e-3)
parser.add_argument('--max_kl', type=int, default=1e-2)
parser.add_argument('--damping', type=int, default=1e-1)
parser.add_argument('--batch_size', type=int, default=10000)
parser.add_argument('--max_episode', type=int, default=100)

cfg = parser.parse_args()


class running_state:
  def __init__(self, state):
    self.len = 1
    self.running_mean = state
    self.running_std = state ** 2

  def update(self, state):
    self.len += 1
    old_mean = self.running_mean.copy()
    self.running_mean[...] = old_mean + (state - old_mean) / self.len
    self.running_std[...] = self.running_std + (state - old_mean) * (state - self.running_mean)

  def mean(self):
    return self.running_mean

  def std(self):
    return np.sqrt(self.running_std / (self.len - 1))


def apply_flat_params(model, flat_params):
  prev_ind = 0
  for param in model.parameters():
    flat_size = int(np.prod(list(param.size())))
    param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
    prev_ind += flat_size


# ------------------------------------Critic-------------------------------------------

class Critic(nn.Module):
  def __init__(self):
    super(Critic, self).__init__()
    self.fc1 = nn.Linear(2, 64)
    self.fc2 = nn.Linear(64, 64)
    self.fc_out = nn.Linear(64, 1)

  def forward(self, x):
    out = F.tanh(self.fc1(x))
    out = F.tanh(self.fc2(out))
    state_value = self.fc_out(out)
    return state_value.squeeze()


def get_state_value(state):
  state_value = critic(state)
  return state_value


# given the states and targets, calculate critic loss under different critic parameters
def critic_loss_fn(states, targets):
  def loss_fn(flat_params):
    apply_flat_params(critic, torch.tensor(flat_params))
    critic.zero_grad()

    state_values = critic(states)
    critic_loss = F.mse_loss(state_values, targets)

    # weight decay
    for param in critic.parameters():
      critic_loss += torch.sum(param ** 2) * cfg.critic_wd

    critic_loss.backward()
    critic_loss = critic_loss.detach().cpu().double().numpy()

    loss_grad_flat = torch.cat([param.grad.view(-1) for param in critic.parameters()])
    loss_grad_flat = loss_grad_flat.detach().cpu().double().numpy()

    return critic_loss, loss_grad_flat

  return loss_fn


def update_critic(states, targets):
  critic_params_flat = torch.cat([param.data.view(-1) for param in critic.parameters()])
  # optimize using the L-BFGS algorithm
  flat_params, _, opt_info = \
    opt.fmin_l_bfgs_b(critic_loss_fn(states, targets), critic_params_flat.cpu().double().numpy(), maxiter=25)
  apply_flat_params(critic, torch.Tensor(flat_params))


# -------------------------------------Actor-------------------------------------------

class Actor(nn.Module):
  def __init__(self):
    super(Actor, self).__init__()
    self.fc1 = nn.Linear(2, 64)
    self.fc2 = nn.Linear(64, 64)
    self.fc_out = nn.Linear(64, 1)
    self.action_sigma = nn.Parameter(torch.ones(1, 1))

  def forward(self, x):
    x = F.tanh(self.fc1(x))
    x = F.tanh(self.fc2(x))
    action_mu = self.fc_out(x)
    return action_mu.squeeze(), self.action_sigma.expand_as(action_mu).squeeze()


def get_action(state):
  action_mu, action_sigma = actor(state)
  action_dist = torch.distributions.Normal(action_mu, action_sigma)
  action = action_dist.sample()
  return action.item()


# given the states, actions, last_log_probs and advantages,
# calculate actor loss (policy gradient with importance sampling) under different actor parameters
def actor_loss_fn(states, actions, last_log_probs, advantages):
  def loss_fn(flat_params=None):
    if flat_params is not None:
      apply_flat_params(actor, flat_params)
    action_mus, action_sigmas = actor(states)
    action_dist = torch.distributions.Normal(action_mus, action_sigmas)
    action_log_probs = action_dist.log_prob(actions)
    action_loss = torch.mean(-advantages * torch.exp(action_log_probs - last_log_probs))
    return action_loss

  return loss_fn


# the kl divergence will always be 0, but the 2nd order gradients are non-zero
def get_kl_divergence(states):
  action_mus, action_sigmas = actor(states)
  fixed_mus, fixed_sigmas = action_mus.detach(), action_sigmas.detach()

  kl_divergence = torch.log(action_sigmas / fixed_sigmas) + \
                  (fixed_sigmas ** 2 + (fixed_mus - action_mus) ** 2) / (2.0 * action_sigmas ** 2) - 0.5
  return torch.mean(kl_divergence)


def Hessian_vector_product(states, vector):
  kl_div = get_kl_divergence(states)

  kl_grads = torch.autograd.grad(kl_div, actor.parameters(), create_graph=True)
  kl_grad_flat = torch.cat([grad.view(-1) for grad in kl_grads])

  kl_grad_vec_prod = kl_grad_flat.dot(vector)
  hess_vec_prod = torch.autograd.grad(kl_grad_vec_prod, actor.parameters())
  hess_vec_prod_flat = torch.cat([grad.contiguous().view(-1) for grad in hess_vec_prod]).data

  return hess_vec_prod_flat + vector * cfg.damping


def conjugate_gradients(states, loss_grad, nsteps, max_error=1e-10):
  x = torch.zeros(loss_grad.size()).cuda()
  r = -loss_grad.clone()
  p = -loss_grad.clone()
  rr = r.dot(r)
  for i in range(nsteps):
    Ap = Hessian_vector_product(states, p)
    alpha = rr / p.dot(Ap)
    x += alpha * p
    r -= alpha * Ap
    new_rr = r.dot(r)
    beta = new_rr / rr
    p = r + beta * p
    rr = new_rr
    if rr < max_error:
      break
  return x


def linesearch(actor_loss_fn, params_flat, fullstep, expected_improve_rate, max_decay_steps=10, accept_ratio=0.1):
  loss = actor_loss_fn().item()
  print("loss before", loss)
  for step in 0.5 ** np.arange(max_decay_steps):
    params_new = params_flat + (step * fullstep).cuda()
    loss_new = actor_loss_fn(params_new).item()
    actual_improve = loss - loss_new
    expected_improve = expected_improve_rate * step
    ratio = actual_improve / expected_improve
    print("a/e/r", actual_improve, expected_improve.item(), ratio.item())

    if ratio > accept_ratio and actual_improve > 0:
      print("loss after", loss_new)
      return True, params_new
  return False, params_flat


def update_actor(states, actions, last_log_probs, advantages):
  loss_fn = actor_loss_fn(states, actions, last_log_probs, advantages)
  loss = loss_fn()
  grads = torch.autograd.grad(loss, actor.parameters())
  loss_grad_flat = torch.cat([grad.view(-1) for grad in grads]).detach()

  step_direction = conjugate_gradients(states, loss_grad_flat, nsteps=10)

  # maximum step size
  beta = torch.sqrt(2 * cfg.max_kl / step_direction.dot(Hessian_vector_product(states, step_direction))).item()

  fullstep = step_direction * beta

  neggdotstepdir = -loss_grad_flat.dot(step_direction)
  print(("lagrange multiplier:", beta, "grad_norm:", loss_grad_flat.norm().item()))
  params_flat = torch.cat([param.data.view(-1) for param in actor.parameters()])
  success, params_new = linesearch(loss_fn, params_flat, fullstep, neggdotstepdir / beta)
  apply_flat_params(actor, params_new)
  return loss.item()


# --------------------------------------Training---------------------------------------

env = gym.make('MountainCarContinuous-v0')
actor = Actor().cuda()
critic = Critic().cuda()


def main():
  state = env.reset()
  state_stat = running_state(state)

  for i in range(cfg.max_episode):
    start_time = time.perf_counter()
    episode_score = 0
    episode = 0
    memory = []

    while len(memory) < cfg.batch_size:
      episode += 1
      state = env.reset()
      state_stat.update(state)
      state = np.clip((state - state_stat.mean()) / (state_stat.std() + 1e-6), -10., 10.)

      for s in range(1000):
        action = get_action(torch.tensor(state).float().cuda()[None, :])
        next_state, reward, done, _ = env.step([action])

        state_stat.update(next_state)
        next_state = np.clip((next_state - state_stat.mean()) / (state_stat.std() + 1e-6), -10., 10.)
        memory.append([state, action, reward, next_state, done])

        state = next_state
        episode_score += reward

        if done:
          break

    state_batch, \
    action_batch, \
    reward_batch, \
    next_state_batch, \
    done_batch = map(lambda x: np.array(x).astype(np.float32), zip(*memory))
    values = get_state_value(state_batch).detach().cpu().numpy()

    returns = np.zeros(action_batch.size(0))
    deltas = np.zeros(action_batch.size(0))
    advantages = np.zeros(action_batch.size(0))

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(reward_batch.size(0))):
      returns[i] = reward_batch[i].item() + cfg.gamma * prev_return * (1 - done_batch[i].item())
      # 计算当前状态的折扣return和当前状态value的差，即GAE中的delta
      deltas[i] = reward_batch[i].item() + cfg.gamma * prev_value * (1 - done_batch[i].item()) - values[i].item()
      # 计算A^GAE来近似真正的advantage，gae_lambda就是GAE中的lambda
      advantages[i] = deltas[i].item() + cfg.gamma * cfg.gae_lambda * prev_advantage * (1 - done_batch[i].item())

      prev_return = returns[i].item()
      prev_value = values[i].item()
      prev_advantage = advantages[i].item()

    advantages = (advantages - advantages.mean()) / advantages.std()

    # using discounted reward as target q-value to update critic
    update_critic(state_batch, returns)

    action_mus, action_sigmas = actor(state_batch)
    action_dist = torch.distributions.Normal(action_mus, action_sigmas)
    action_log_probs = action_dist.log_prob(action_batch).detach()

    update_actor(state_batch, action_batch, action_log_probs, advantages)

    episode_score /= episode
    print(',last_score {:8f}, steps {}, ({:2f} sec/eps)'.
          format(episode_score, len(memory), time.perf_counter() - start_time))
    avg_score_plot.append(avg_score_plot[-1] * 0.99 + episode_score * 0.01)
    last_score_plot.append(episode_score)
    drawnow(draw_fig)

  env.close()


if __name__ == '__main__':
  main()
  plt.pause(0)
