import numpy as np
from Policies.MLP_policy import MLPPolicy
from Critics.v_critic import VanillaCritic
class ACAgent(object):
    def __init__(self, args, **kwargs):
        super(ACAgent, self).__init__(**kwargs)

        self.gamma = args.gamma
        self.actor = MLPPolicy(args)
        self.critic = VanillaCritic(args)

    def advantage_td0(self, obs, next_obs, rews, terminals):

        adv_n = rews + self.gamma * self.critic.critic_prediction(next_obs).numpy() * np.logical_not(terminals) - self.critic.critic_prediction(obs).numpy()
        return adv_n

    def train(self, obs, acs, rews, next_obs, terminals):
        advs = self.advantage_td0(obs, next_obs, rews, terminals)
        critic_loss = self.critic.update(obs, next_obs, rews, terminals)
        actor_loss = self.actor.update(obs, acs, advs)
        return critic_loss, actor_loss

    def add_to_replay_buffer(self, paths):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError