from Agents.base_agent import BaseAgent
from Policies.random_policy import RandomPolicy
import numpy as np

class QLAgent(BaseAgent):
    def __init__(self, args, env):
        self.env = env
        self.lr = args.learning_rate
        self.gamma = args.gamma
        self.q_table = np.zeros(16 * 4).reshape(16, 4).astype(np.float32)
        self.actor = RandomPolicy(env.action_space)

    def train(self, obs, acs, rews, next_obs, terminals):
        for i in range(len(obs)):
            q_next = np.max(self.q_table[next_obs[i]])
            self.q_table[obs[i]][acs[i]] = (1 - self.lr) * self.q_table[obs[i]][acs[i]] \
                + self.lr * (rews[i] + self.gamma * q_next)
    
    def get_action(self, ob):
        return np.argmax(self.q_table[ob])
