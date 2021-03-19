from Agents.pg_agent import PGAgent
from Agents.ac_agent import ACAgent
from Agents.q_learning_agent import QLAgent
from Policies.MLP_policy import MLPPolicy
import time, random, os
import gym
import numpy as np

from Policies.random_policy import RandomPolicy
import utils.utils as utils

class Trainer(object):

    def __init__(self, args):

        # Make the gym environment
        self.env = gym.make(args.env)
        self.env.seed(args.seed)
        random.seed(args.seed)

        # Num of iterations
        self.train_iter = args.train_iter
        self.test_iter = args.test_iter

        args.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        args.ac_dim = utils.get_space_dim(self.env.action_space)
        args.ob_dim = utils.get_space_dim(self.env.observation_space)

        # Make agent
        self.agent = QLAgent(args)
        expert_path = os.path.join('.', 'Experts', 'QLearning', 'expert.npy')
        self.agent.load(expert_path)

        # Training Trajectory Collect Policy
        if args.collect_policy == 'random':
            self.collect_policy = RandomPolicy(self.env.action_space)
        else:
            self.collect_policy = self.agent.actor

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
        for itr in range(self.test_iter):
            obs, acs, rewards, next_obs, terminals, image_obs = utils.sample_trajectory(self.env, self.agent.actor, 200, True, render_mode=())
            ep_rewards.append(np.sum(rewards))

        print("Average Total Rewards: {}".format(np.mean(ep_rewards)))


    def logging(self, itr, rewards):
        print('Rewards: {}'.format(np.sum(rewards)))        
        print('EpLen: {}'.format(len(rewards)))

