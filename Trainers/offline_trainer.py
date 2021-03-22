from Agents.pg_agent import PGAgent
from Agents.ac_agent import ACAgent
from Agents.q_learning_agent import QLAgent
from Policies.MLP_policy import MLPPolicy
import time, random, os
import gym
import numpy as np
import matplotlib.pyplot as plt

from Policies.random_policy import RandomPolicy
from ReplayBuffers.offline_buffer import OfflineBuffer
import utils.utils as utils

class OfflineTrainer(object):

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

        # Make expert
        self.expert = QLAgent(args)
        expert_path = os.path.join('.', 'Experts', 'QLearning', 'expert.npy')
        self.expert.load(expert_path)
    
        # Offline buffer
        self.buffer = OfflineBuffer(args.buffer_size)
        self.batch_size = args.batch_size

        # Training Trajectory Collect Policy
        if args.collect_policy == 'random':
            self.collect_policy = RandomPolicy(self.env.action_space)
        else:
            self.collect_policy = self.expert.actor
    
    def generate_buffer(self):
        print("Generating offline dataset...")
        while not self.buffer.full():
            obs, acs, rewards, next_obs, terminals, image_obs = utils.sample_trajectory(self.env, self.collect_policy, 200, True, render_mode=())
            self.buffer.add_trajectory(utils.Path(obs, image_obs, acs, rewards, next_obs, terminals))
        print("Offline dataset Generated")


    def train(self):
        self.agent.training()
        for itr in range(self.train_iter):
            print('************ Iteration {} ************'.format(itr))
            obs, acs, rewards, next_obs, terminals = self.buffer.sample_random_data(self.batch_size)
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
        plt.savefig('./img2.png')
        plt.close()

        print("Average Total Rewards: {}".format(np.mean(ep_rewards)))
        if self.save_model:
            expert_dir = os.path.join('.', 'Experts', 'Offline')
            self.agent.save(expert_dir)


    def logging(self, itr, rewards):
        print('Rewards: {}'.format(np.sum(rewards)))        
        print('EpLen: {}'.format(len(rewards)))

