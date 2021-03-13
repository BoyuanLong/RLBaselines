from numpy.core.numeric import indices
from Policies.MLP_policy import MLPPolicy
from Agents.gae import discounted_cumsum


class PGAgent(object):
    def __init__(self, args, **kwargs):
        super(PGAgent, self).__init__()

        self.actor = MLPPolicy(args)
        self.gamma = args.gamma

    def train(self, obs, acs, rews, next_obs, terminals):
        adv = discounted_cumsum(self.gamma, rews)
        loss = self.actor.update(obs, acs, adv)
        return loss

    def add_to_replay_buffer(self, paths):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError