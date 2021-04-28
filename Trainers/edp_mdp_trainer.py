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
        # self.expert = QLAgent(args)
        # expert_path = os.path.join('.', 'Experts', 'QLearning', 'expert.npy')
        # self.expert.load(expert_path)
    
        # Offline buffer
        # self.buffer = OfflineBuffer(args.buffer_size)
        # self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.concat_rewards = args.concat_rewards
        # Replay Buffer
        self.buffer = ReplayBuffer(args.buffer_size)

        # Training Trajectory Collect Policy
        if args.collect_policy == 'random':
            self.collect_policy = RandomPolicy(self.env.action_space)
        else:
            self.collect_policy = self.agent.actor

        # Logger
        self.logger = Logger(args.logdir)
    
    # def generate_buffer(self):
    #     print("Generating offline dataset...")
    #     counter = 0
    #     data = defaultdict(lambda : defaultdict(list))
    #     while counter < self.buffer_size:
    #         path = utils.sample_trajectory(self.env, self.collect_policy, 200, True, render_mode=())
    #         obs, acs, rewards, next_obs, terminals = path["observation"], path["action"], path["reward"], path["next_observation"], path["terminal"]
    #         # assert len(obs) == len(acs) == len(rewards) == len(terminals) == len(next_obs)
    #         print(len(obs), len(acs), len(rewards), len(terminals), len(next_obs))
    #         for i in range(len(obs)):
    #             s = obs[i]
    #             a = acs[i]
    #             data[s][a].append((rewards[i], next_obs[i], terminals[i]))
    #         counter += len(obs)
    #     print("Offline dataset Generated")
    #     self.data = data


    def train(self):

        self.agent.training()
        for itr in range(self.train_iter):
            print('************ Iteration {} ************'.format(itr))
            paths, _ = utils.sample_trajectories(self.env, self.collect_policy, self.batch_size, 50, True, render_mode=())
            for p in paths:
                p['observation'] = p['observation'].astype(np.int)
                p['action'] = p['action'].astype(np.int)
                p['next_observation'] = p['next_observation'].astype(np.int)

            # print(p)
            self.buffer.add_trajectory(paths)

            observations, actions, unconcatenated_rews, next_observations, terminals = self.buffer.sample_recent_data(self.batch_size, concat_rew=True)
            log = self.agent.train(observations, actions, unconcatenated_rews, next_observations, terminals)

            self.logging(itr, paths, log)
    
    def test(self):
        self.agent.testing()
        ep_rewards = []
        state_feq = np.zeros(self.ob_dim)
        for itr in range(self.test_iter):
            path = utils.sample_trajectory(self.env, self.agent.actor, 50, True, render_mode=())
            rewards = path["reward"]
            ep_rewards.append(np.sum(rewards))

        if self.save_model:
            expert_dir = os.path.join('.', 'Experts', 'MDP')
            if not os.path.exists(expert_dir):
                os.makedirs(expert_dir)
            self.agent.save(expert_dir)


    def logging(self, itr, train_paths, agent_log):
        epislon = self.agent.testing()
        eval_paths, _ = utils.sample_trajectories(self.env, self.collect_policy, self.batch_size, 50, False, render_mode=())

        # if itr % 20 == 0:
        #     _ = utils.sample_n_trajectories(self.env, self.collect_policy, 5, 200, False, render_mode=())

        train_returns = [path["reward"].sum() for path in train_paths]
        eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

        train_ep_lens = [len(path["reward"]) for path in train_paths]
        eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

        logs = OrderedDict()
        logs["Eval_AverageReturn"] = np.mean(eval_returns)
        logs["Eval_StdReturn"] = np.std(eval_returns)
        logs["Eval_MaxReturn"] = np.max(eval_returns)
        logs["Eval_MinReturn"] = np.min(eval_returns)
        logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

        logs["Train_AverageReturn"] = np.mean(train_returns)
        logs["Train_StdReturn"] = np.std(train_returns)
        logs["Train_MaxReturn"] = np.max(train_returns)
        logs["Train_MinReturn"] = np.min(train_returns)
        logs["Train_AverageEpLen"] = np.mean(train_ep_lens)
        if agent_log:
            logs.update(agent_log)

        # logs["Train_EnvstepsSoFar"] = self.total_envsteps
        # logs["TimeSinceStart"] = time.time() - self.start_time

        # perform the logging
        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, itr)
        print('Done logging...\n\n')
        self.agent.training(epislon)
        self.logger.flush()

