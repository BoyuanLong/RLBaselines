from Agents.pg_agent import PGAgent
from Agents.ac_agent import ACAgent
from Agents.q_learning_agent import QLAgent
from Policies.MLP_policy import MLPPolicy
import time, random, os
import gym
import numpy as np
import matplotlib.pyplot as plt

from Policies.random_policy import RandomPolicy
import utils.utils as utils
from utils.logger import Logger
from collections import OrderedDict

class Trainer(object):

    def __init__(self, args):

        # Make the gym environment
        self.env = gym.make(args.env)
        self.env.seed(args.seed)

        self.save_model = args.save_model

        # Num of iterations
        self.train_iter = args.train_iter
        self.test_iter = args.test_iter

        args.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        args.ac_dim = utils.get_space_dim(self.env.action_space)
        args.ob_dim = utils.get_space_dim(self.env.observation_space)
        self.ob_dim = args.ob_dim
        self.ac_dim = args.ac_dim

        # Make agent
        self.agent = QLAgent(args)

        # Training Trajectory Collect Policy
        if args.collect_policy == 'random':
            self.collect_policy = RandomPolicy(self.env.action_space)
        else:
            self.collect_policy = self.agent.actor

        # Logger
        self.logger = Logger(args.logdir)

    def train(self):

        self.agent.training()
        for itr in range(self.train_iter):
            print('************ Iteration {} ************'.format(itr))
            obs, acs, rewards, next_obs, terminals, image_obs = utils.sample_trajectory(self.env, self.collect_policy, 200, True, render_mode=())
            obs = np.array(obs)
            acs = np.array(acs)
            rewards = np.array(rewards)
            next_obs = np.array(next_obs)
            terminals = np.array(terminals)
            loss = self.agent.train(obs, acs, rewards, next_obs, terminals)

            self.logging(itr, rewards)
    
    def test(self):
        self.agent.testing()
        ep_rewards = []
        state_feq = np.zeros(self.ob_dim)
        for itr in range(self.test_iter):
            obs, acs, rewards, next_obs, terminals, image_obs = utils.sample_trajectory(self.env, self.agent.actor, 200, True, render_mode=())
            ep_rewards.append(np.sum(rewards))

            for i in range(len(obs)):
                state_feq[obs[i]] += 1
                if terminals[i]:
                    state_feq[next_obs[i]] += 1

        plt.bar(list(range(self.ob_dim)), state_feq, log=True)
        plt.savefig('./img.png')
        plt.close()

        print("Average Total Rewards: {}".format(np.mean(ep_rewards)))
        if self.save_model:
            expert_dir = os.path.join('.', 'Experts')
            self.agent.save(expert_dir)


    def logging(self, itr, rewards):
        print('Rewards: {}'.format(np.sum(rewards)))        
        print('EpLen: {}'.format(len(rewards)))

        logs = OrderedDict()
        logs['Return'] = np.sum(rewards)
        logs['EpLen'] = len(rewards)
        # perform the logging
        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, itr)
        print('Done logging...\n\n')
        self.logger.flush()

