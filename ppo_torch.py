import numpy as np
import gymnasium as gym
import time
from gymnasium.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

def Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError
    
    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError
    
    def forward(self, obs, act=None):
        # create action distribution for a given obs
        # also compute log likelihood of actors
        pi = self._distribution(obs)
        logp_a = None

        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pong-v0')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--kl', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--from_pixel', action="store_true")
    parser.add_argument('--render', action="store_true")
    parser.add_argument('--load_from', action="store_true")
    args = parser.parse_args()

    # ppo(lambda : gym.make(args.env, render_mode='human' if args.render else None), actor_critic=ActorCritic,
    #     ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), target_kl=args.kl, gamma=args.gamma, 
    #     seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs, load_from=args.load_from, device=torch.device(args.device))
