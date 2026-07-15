import numpy as np
import gymnasium as gym
import time
from gymnasium.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


# building blocks
def mlp(sizes, activation, output_activation=nn.Identity): # use identity to get the last layer's output so we can handle it as needed later

    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation #for last layer we do not apply activation function
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def cnn(in_channels, activation=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=8, stride=4),
        activation(),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
        activation(),
        nn.Flatten()
    )

# actor
class Actor(nn.Module):
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

class CategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, cnn=None):
        super().__init__()
        self.cnn = cnn
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        if self.cnn:
            obs = self.cnn(obs)
        logits = self.logits_net(obs)
        return Categorical(logits=logits)
    
    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)    

class GaussianActor(Actor):
    pass

# critic
class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation, cnn=None):
        super().__init__()
        self.cnn = cnn
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)  # last layer is a single value for the value function

    def forward(self, obs):
        if self.cnn:
            obs = self.cnn(obs)
        return torch.squeeze(self.v_net(obs), -1) # change shape to (batch_size,)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_space, hidden_sizes=(64,64), activation=nn.ReLU, cnn_enable=False, frames=3):
        super().__init__()
        
        if cnn_enable:
            cn_net = cnn(frames, activation)
        else:
            cn_net = None
        
        # policy based on continuous or discrete action space
        if isinstance(action_space, Box):
            self.pi = GaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation, cn_net)
        elif isinstance(action_space, Discrete):
            self.pi = CategoricalActor(obs_dim, action_space.n, hidden_sizes, activation, cn_net)
        else:
            raise ValueError(f"Invalid action space: {action_space}")
        
        self.v = Critic(obs_dim, hidden_sizes, activation, cn_net)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()


# def ppo(env_fn, actor_critic=ActorCritic, ac_kwargs=dict(), seed=0, 
#         steps_per_epoch=6000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
#         vf_lr=3e-4, train_pi_iters=10, train_v_iters=10, lam=0.97, max_ep_len=6000,
#         target_kl=0.1, save_freq=10, load_from=None, device='cpu'):
    
#     print(locals())
#     print(open(__file__).read())

#     # Random seed
#     seed = 42
#     torch.manual_seed(seed)
#     np.random.seed(seed)




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
