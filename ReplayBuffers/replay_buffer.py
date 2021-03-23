import numpy as np
from numpy.lib.arraysetops import isin
import utils.utils as utils

class ReplayBuffer(object):

    def __init__(self, max_size=1000000):

        self.max_size = max_size
        self.paths = []
        self.obs = None
        self.acs = None
        self.rews = None
        self.rew_lists = None
        self.next_obs = None
        self.terminals = None

    def size(self):
        return len(self.obs)

    def full(self):
        return len(self.obs) == self.max_size if self.obs is not None else False

    def add_trajectory(self, paths):
        assert isinstance(paths, list)
        self.paths.extend(paths)

        observations, actions, next_observations, terminals, concat_rewards, unconcat_rewards = utils.convert_listofrollouts(paths)

        if self.obs is None:
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
            self.rews = concat_rewards[-self.max_size:]
            self.rew_lists = unconcat_rewards[-self.max_size:]
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
        else:
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
            self.next_obs = np.concatenate([self.next_obs, next_observations])[-self.max_size:]
            self.terminals = np.concatenate([self.terminals, terminals])[-self.max_size:]
            self.rews = np.concatenate([self.rews, concat_rewards])[-self.max_size:]
            self.rew_lists = (self.rew_lists + unconcat_rewards)[-self.max_size:]


    ########################################
    ########################################

    def sample_random_rollouts(self, num_rollouts):
        rand_indices = np.random.permutation(len(self.paths))[:num_rollouts]
        return self.paths[rand_indices]

    def sample_recent_rollouts(self, num_rollouts=1):
        return self.paths[-num_rollouts:]

    ########################################
    ########################################

    def sample_random_data(self, batch_size):

        assert self.obs.shape[0] == self.acs.shape[0] == self.rews.shape[0] == self.next_obs.shape[0] == self.terminals.shape[0]
        rand_indices = np.random.permutation(self.obs.shape[0])[:batch_size]
        return self.obs[rand_indices], self.acs[rand_indices], self.rews[rand_indices], self.next_obs[rand_indices], self.terminals[rand_indices]

    def sample_recent_data(self, batch_size=1, concat_rew=True):

        if concat_rew:
            return self.obs[-batch_size:], self.acs[-batch_size:], self.rews[-batch_size:], self.next_obs[-batch_size:], self.terminals[-batch_size:]
        else:
            num_recent_rollouts_to_return = 0
            num_datapoints_so_far = 0
            index = -1
            while num_datapoints_so_far < batch_size:
                recent_rollout = self.paths[index]
                index -= 1
                num_recent_rollouts_to_return += 1
                num_datapoints_so_far += utils.get_pathlength(recent_rollout)
            rollouts_to_return = self.paths[-num_recent_rollouts_to_return:]
            observations, actions, next_observations, terminals, concatenated_rews, unconcatenated_rews = utils.convert_listofrollouts(rollouts_to_return)
            return observations, actions, unconcatenated_rews, next_observations, terminals

