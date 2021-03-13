from Agents.pg_agent import PGAgent
from Agents.ac_agent import ACAgent
from Policies.MLP_policy import MLPPolicy
import time
import gym
import numpy as np

from Policies.random_policy import RandomPolicy
import utils.utils as utils

class Trainer(object):

    def __init__(self, args):

        # Make the gym environment
        self.env = gym.make(args.env)
        self.env.seed(args.seed)

        self.n_iter = args.n_iter

        args.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        args.ac_dim = self.env.action_space.n if args.discrete else 0
        args.ob_dim = self.env.observation_space.shape[0]

        # Make agent
        self.agent = ACAgent(args)

    def train(self):
        for itr in range(self.n_iter):
            print('************ Iteration {} ************'.format(itr))
            obs, acs, rewards, next_obs, terminals, image_obs = utils.sample_trajectory(self.env, self.agent.actor, 200, True)
            obs = np.array(obs)
            acs = np.array(acs)
            rewards = np.array(rewards)
            next_obs = np.array(next_obs)
            terminals = np.array(terminals)
            loss = self.agent.train(obs, acs, rewards, next_obs, terminals)

            self.logging(itr, rewards)

    def logging(self, itr, rewards):
        print('Rewards: {}'.format(np.sum(rewards)))        
        print('EpLen: {}'.format(len(rewards)))

