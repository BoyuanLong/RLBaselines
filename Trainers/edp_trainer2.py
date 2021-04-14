from typing import DefaultDict

from numpy.lib.polynomial import RankWarning
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

from Policies.random_policy import RandomPolicy
from ReplayBuffers.offline_buffer import OfflineBuffer
import utils.utils as utils

class EDPTrainer(object):

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
        self.agent.actor.q_table = np.random.rand(self.ob_dim, self.ac_dim).astype(np.float32)
        print(self.agent.actor.q_table)
        print("====")

        # Make expert
        self.expert = QLAgent(args)
        expert_path = os.path.join('.', 'Experts', 'QLearning', 'expert.npy')
        self.expert.load(expert_path)
    
        # Offline buffer
        self.buffer = OfflineBuffer(args.buffer_size)
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size

        # Training Trajectory Collect Policy
        if args.collect_policy == 'random':
            self.collect_policy = RandomPolicy(self.env.action_space)
        else:
            self.collect_policy = self.expert.actor
    
    def generate_buffer(self):
        print("Generating offline dataset...")
        counter = 0
        data = defaultdict(lambda : defaultdict(list))
        while counter < self.buffer_size:
            path = utils.sample_trajectory(self.env, self.collect_policy, 200, True, render_mode=())
            obs, acs, rewards, next_obs, terminals = path["observation"], path["action"], path["reward"], path["next_observation"], path["terminal"]
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
            values = defaultdict(lambda: 0)
            for s, v in self.data.items():
                for i in range(100000):
                    # if s == 23.0:
                    #     print("====")
                    obs = []
                    rewards = []
                    ob = s
                    step = 0
                    total_rewards = 0
                    while True:
                        a = np.float32(self.agent.get_action(ob.astype(int)))
                        if ob in [31.0, 39.0, 47.0, 55.0]:
                            a = 2.0
                            # print(len(self.data[ob][2.]))
                        if s in [23.0]:
                            # print(a, ob)
                            # for k23, v23 in self.data[ob].items():
                            #     print(k23, len(v23))
                            # a = 2.0
                            obs.append(ob)

                        if len(self.data[ob][a]) != 0:
                            a_data = self.data[ob][a]
                            index = np.random.randint(len(a_data))
                            # if s == 23.0:
                            #     print(ob, a)
                            r, ob, done = a_data[index]
                            # if s == 23.0:
                            #     print(r, ob, done)
                            #     print(step)
                            rewards.append(r)
                            total_rewards += r

                            if done or step > 300:
                                break
                            step += 1
                        else:
                            break
                    values[s] += total_rewards
            print(values)
            
            q_values = np.zeros(self.ob_dim * self.ac_dim).reshape(self.ob_dim, self.ac_dim).astype(np.float32) 
            for s, v_data in self.data.items():
                for action, action_data in v_data.items():
                    rand_indices = np.random.permutation(len(action_data))[:self.batch_size]
                    r = []
                    for i in rand_indices:
                        val = 0 if action_data[i][2] else values[action_data[i][1]]
                        r.append(action_data[i][0] + val)
                    q_values[s.astype(int)][action.astype(int)] = 0 if len(r) == 0 else np.mean(r)
            self.agent.actor.q_table = q_values
            self.agent.actor.epsilon = max(self.agent.actor.epsilon - self.agent.actor.e_decay_rate, 0.0)
            print(self.agent.actor.q_table)

            self.logging(itr, [])
    
    def test(self):
        self.agent.testing()
        ep_rewards = []
        state_feq = np.zeros(self.ob_dim)
        for itr in range(self.test_iter):
            path = utils.sample_trajectory(self.env, self.agent.actor, 200, True, render_mode=())
            rewards = path["reward"]
            ep_rewards.append(np.sum(rewards))

        print("Average Total Rewards: {}".format(np.mean(ep_rewards)))
        if self.save_model:
            expert_dir = os.path.join('.', 'Experts', 'Offline')
            self.agent.save(expert_dir)


    def logging(self, itr, rewards):
        print('Rewards: {}'.format(np.sum(rewards)))        
        print('EpLen: {}'.format(len(rewards)))

