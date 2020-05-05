"""
This file is filled with miscelaneous classes and functions.
"""
import gym
from gym.wrappers import AtariPreprocessing, TransformReward

import torch
import numpy as np

from torch.distributions import Categorical


class ReturnWrapper(gym.Wrapper):
    #######################################################################
    # Copyright (C) 2020 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
    # Permission given to modify the code as long as you keep this        #
    # declaration at the top                                              #
    #######################################################################
    def __init__(self, env):
        super().__init__(env)
        self.total_rewards = 0
        self.steps = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_rewards += reward
        self.steps += 1
        if done:
            info['returns/episodic_reward'] = self.total_rewards
            info['returns/episodic_length'] = self.steps
            self.total_rewards = 0
            self.steps = 0
        else:
            info['returns/episodic_reward'] = None
            info['returns/episodic_length'] = None
        return obs, reward, done, info


def basic_wrapper(env):
    """Use this as a wrapper only for cartpole etc."""
    env = ReturnWrapper(env)
    env = TransformReward(env, lambda r: np.clip(r, -1, 1))
    return env


def atari_wrapper(env):
    # This is substantially the same CNN as in (Mnih et al., 2016; 2015),
    # the only difference is that in the pre-processing stage
    # we retain all colour channels.
    env = AtariPreprocessing(env, grayscale_obs=False, scale_obs=True)
    env = ReturnWrapper(env)
    env = TransformReward(env, lambda r: np.sign(r))
    return env


def make_envs(env_name, num_envs, seed=0):
    env_ = gym.make(env_name)
    is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env_.unwrapped, gym.envs.atari.atari_env.AtariEnv)

    if is_atari:
        wrapper_fn = atari_wrapper
    else:
        wrapper_fn = basic_wrapper

    envs = gym.vector.make(env_name, num_envs, wrappers=wrapper_fn)
    envs.seed(seed)
    return envs


def take_action(a):
    dist = Categorical(a)
    action = dist.sample()
    logp = dist.log_prob(action)
    entropy = dist.entropy()
    return action.cpu().detach().numpy(), logp, entropy


def init_hidden(n_workers, h_dim, device, grad=False):
    return (torch.zeros(n_workers, h_dim, requires_grad=grad).to(device),
            torch.zeros(n_workers, h_dim, requires_grad=grad).to(device))


def init_obj(n_workers, h_dim, c, device):
    goals = [torch.zeros(n_workers, h_dim, requires_grad=True).to(device)
             for _ in range(c)]
    states = [torch.zeros(n_workers, h_dim).to(device) for _ in range(c)]
    return goals, states


def weight_init(layer):
    if type(layer) == torch.nn.modules.conv.Conv2d or \
            type(layer) == torch.nn.Linear:
        torch.nn.init.orthogonal_(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias.data, 0)

