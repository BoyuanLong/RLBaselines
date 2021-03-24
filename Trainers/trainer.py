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
from ReplayBuffers.replay_buffer import ReplayBuffer


AGENTS = {
    'ac': ACAgent,
    'reinforce': PGAgent,
}
class Trainer(object):

    def __init__(self, args):

        # Make the gym environment
        self.env = gym.make(args.env)
        self.env.seed(args.seed)

        self.save_model = args.save_model

        # Num of iterations
        self.train_iter = args.train_iter
        self.test_iter = args.test_iter
        self.batch_size = args.batch_size

        args.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        args.ac_dim = utils.get_space_dim(self.env.action_space)
        args.ob_dim = utils.get_space_dim(self.env.observation_space)
        self.ob_dim = args.ob_dim
        self.ac_dim = args.ac_dim

        # Make agent
        agent_class = AGENTS[args.agent]
        self.agent = agent_class(args)

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

    def train(self):

        self.agent.training()
        for itr in range(self.train_iter):
            print('************ Iteration {} ************'.format(itr))
            paths, _ = utils.sample_trajectories(self.env, self.collect_policy, self.batch_size, 200, True, render_mode=())
            self.buffer.add_trajectory(paths)

            observations, actions, unconcatenated_rews, next_observations, terminals = self.buffer.sample_recent_data(self.batch_size, concat_rew=self.concat_rewards)
            log = self.agent.train(observations, actions, unconcatenated_rews, next_observations, terminals)

            self.logging(itr, paths, log)
    
    def test(self):
        self.agent.testing()
        ep_rewards = []
        for itr in range(self.test_iter):
            paths = utils.sample_trajectory(self.env, self.agent.actor, 200, True, render_mode=())
            # ep_rewards.append(np.sum(rewards))

        # print("Average Total Rewards: {}".format(np.mean(ep_rewards)))
        if self.save_model:
            expert_dir = os.path.join('.', 'Experts')
            self.agent.save(expert_dir)


    def logging(self, itr, train_paths, agent_log):
        eval_paths, _ = utils.sample_trajectories(self.env, self.collect_policy, self.batch_size, 200, True, render_mode=())

        if itr % 20 == 0:
            _ = utils.sample_n_trajectories(self.env, self.collect_policy, 5, 200, True, render_mode=('human'))

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
        logs.update(agent_log)

        # logs["Train_EnvstepsSoFar"] = self.total_envsteps
        # logs["TimeSinceStart"] = time.time() - self.start_time

        # perform the logging
        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, itr)
        print('Done logging...\n\n')
        self.logger.flush()

