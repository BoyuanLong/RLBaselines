from collections import OrderedDict
import numpy as np
from Policies.MLP_policy import MLPPolicy
from Critics.v_critic import VanillaCritic
class ACAgent(object):
    def __init__(self, args, **kwargs):
        super(ACAgent, self).__init__(**kwargs)

        self.gamma = args.gamma
        self.actor = MLPPolicy(args)
        self.critic = VanillaCritic(args)

        self.istraining = False
    
    def training(self):
        self.istraining = True
    
    def testing(self):
        self.istraining = False

    def advantage_td0(self, obs, next_obs, rews, terminals):
        assert obs.shape[0] == next_obs.shape[0] == rews.shape[0] == terminals.shape[0]
        adv_n = rews + self.gamma * self.critic.critic_prediction(next_obs).numpy() * np.logical_not(terminals) - self.critic.critic_prediction(obs).numpy()
        adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
        assert len(adv_n.shape) == 1
        return adv_n

    def train(self, obs, acs, rews, next_obs, terminals):
        advs = self.advantage_td0(obs, next_obs, rews, terminals)
        log = OrderedDict()
        critic_log = self.critic.update(obs, next_obs, rews, terminals)
        actor_loss = self.actor.update(obs, acs, advs)

        log.update(critic_log)
        return log

    def add_to_replay_buffer(self, paths):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError