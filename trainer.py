from Agents.pg_agent import PGAgent
from Agents.ac_agent import ACAgent
from Agents.q_learning_agent import QLAgent
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
        args.ac_dim = utils.get_space_dim(self.env.action_space)
        args.ob_dim = utils.get_space_dim(self.env.observation_space)

        # Make agent
        self.agent = QLAgent(args, self.env)

    def train(self):
        for itr in range(self.n_iter):
            print('************ Iteration {} ************'.format(itr))
            obs, acs, rewards, next_obs, terminals, image_obs = utils.sample_trajectory(self.env, self.agent.actor, 200, True, render_mode=('human'))
            obs = np.array(obs)
            acs = np.array(acs)
            rewards = np.array(rewards)
            next_obs = np.array(next_obs)
            terminals = np.array(terminals)
            loss = self.agent.train(obs, acs, rewards, next_obs, terminals)

            self.logging(itr, rewards)
    
    def test(self):
        ep_rewards = []
        for itr in range(1000):
            obs, acs, rewards, next_obs, terminals, image_obs = utils.sample_trajectory(self.env, self.agent, 200, True, render_mode=('human'))
            ep_rewards.append(np.sum(rewards))

        print("Average Rewards: {}".format(np.average(ep_rewards)))


    def logging(self, itr, rewards):
        print('Rewards: {}'.format(np.sum(rewards)))        
        print('EpLen: {}'.format(len(rewards)))

