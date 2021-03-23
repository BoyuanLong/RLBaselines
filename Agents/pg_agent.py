from numpy.core.numeric import indices
from Policies.MLP_policy import MLPPolicy
import numpy as np
from Agents.gae import discounted_cumsum


class PGAgent(object):
    def __init__(self, args, **kwargs):
        super(PGAgent, self).__init__()

        self.actor = MLPPolicy(args)
        self.gamma = args.gamma
        self.istraining = False

    def training(self):
        self.istraining = True

    def testing(self):
        self.istraining = False
        
    def train(self, obs, acs, rew_lists, next_obs, terminals):
        adv = np.concatenate([discounted_cumsum(self.gamma, rews) for rews in rew_lists])
        loss = self.actor.update(obs, acs, adv)
        return loss

    def add_to_replay_buffer(self, paths):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError