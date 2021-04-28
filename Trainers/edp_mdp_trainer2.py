from typing import DefaultDict
from Agents.pg_agent import PGAgent
from Agents.ac_agent import ACAgent
from Agents.q_learning_agent import QLAgent
from Policies.MLP_policy import MLPPolicy
import time, random, os
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from utils.plot_utils import heatmap
from Envs.edp_mdp import MDP
from ReplayBuffers.replay_buffer import ReplayBuffer
from utils.logger import Logger
from collections import OrderedDict

from Policies.random_policy import RandomPolicy
from ReplayBuffers.offline_buffer import OfflineBuffer
import utils.utils as utils

class MDPTrainer(object):

    def __init__(self, args):

        # Make the gym environment
        self.env = MDP(10, 5, 'test.npy', 'r_file.npy')
        # self.env = gym.make(args.env)
        # self.env.seed(args.seed)

        self.save_model = args.save_model

        # Num of iterations
        self.train_iter = args.train_iter
        self.test_iter = args.test_iter

        args.discrete = True
        args.ac_dim = 5
        args.ob_dim = 10

        self.ob_dim = args.ob_dim
        self.ac_dim = args.ac_dim

        # Make agent
        self.agent = QLAgent(args)

        # Make expert
        self.expert = QLAgent(args)
        expert_path = os.path.join('.','Experts', 'MDP', 'QLearning', 'expert.npy')
        self.expert.load(expert_path)
    
        # Offline buffer
        self.buffer = OfflineBuffer(args.buffer_size)
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.concat_rewards = args.concat_rewards
        # Replay Buffer
        # self.buffer = ReplayBuffer(args.buffer_size)

        # Training Trajectory Collect Policy
        if args.collect_policy == 'random':
            self.collect_policy = RandomPolicy(self.env.na)
        else:
            self.collect_policy = self.agent.actor

        # Logger
        self.logger = Logger(args.logdir)
    
    def generate_buffer(self):
        print("Generating offline dataset...")
        counter = 0
        data = defaultdict(lambda : defaultdict(list))
        while counter < self.buffer_size:
            path = utils.sample_trajectory(self.env, self.collect_policy, 50, True, render_mode=())
            obs, acs, rewards, next_obs, terminals = path["observation"].astype(np.int), path["action"].astype(np.int), path["reward"], path["next_observation"].astype(np.int), path["terminal"]
            # assert len(obs) == len(acs) == len(rewards) == len(terminals) == len(next_obs)
            print(len(obs), len(acs), len(rewards), len(terminals), len(next_obs))
            for i in range(len(obs)):
                s = obs[i]
                a = acs[i]
                data[s][a].append((rewards[i], next_obs[i], terminals[i]))
            counter += len(obs)
        print("Offline dataset Generated")
        self.data = data


    def train(self):
        self.agent.training()
        for itr in range(self.train_iter):
            print('************ Iteration {} ************'.format(itr))
            for s, v in self.data.items():
                for a, data in v.items():
                    rand_indices = np.random.permutation(len(data))[:self.batch_size]
                    rewards = []
                    next_obs = []
                    terminals = []
                    for i in rand_indices:
                        rewards.append(data[i][0])
                        next_obs.append(data[i][1])
                        terminals.append(data[i][2])

                    obs = np.full(len(rewards), s, dtype=np.int)
                    acs = np.full(len(rewards), a, dtype=np.int)

                    loss = self.agent.train(obs, acs, np.array(rewards, dtype=np.int), np.array(next_obs, dtype=np.int), terminals)

            self.logging(itr, rewards)
    
    def test(self):
        self.agent.testing()
        ep_rewards = []
        state_feq = np.zeros(self.ob_dim)
        for itr in range(self.test_iter):
            path = utils.sample_trajectory(self.env, self.collect_policy, 50, True, render_mode=())
            rewards = path["reward"]
            ep_rewards.append(np.sum(rewards))

        print("Average Total Rewards: {}".format(np.mean(ep_rewards)))
        if self.save_model:
            expert_dir = os.path.join('.', 'Experts', 'Offline')
            self.agent.save(expert_dir)


    def logging(self, itr, rewards):
        print('Rewards: {}'.format(np.sum(rewards)))        
        print('EpLen: {}'.format(len(rewards)))

