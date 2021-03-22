from Agents.base_agent import BaseAgent
from Policies.q_tabular_policy import QPolicy
import os

class QLAgent(BaseAgent):
    def __init__(self, args):
        super().__init__()
        self.is_training = False
        self.actor = QPolicy(args)

    def training(self):
        self.is_training = True
        self.actor.training()

    def testing(self):
        self.is_training = False
        self.actor.testing()

    def train(self, obs, acs, rews, next_obs, terminals):
        self.actor.update(obs, acs, rews, next_obs, terminals)
    
    def get_action(self, ob):
        return self.actor.get_action(ob)
    
    def save(self, filepath):
        filepath = os.path.join(filepath, 'QLearning')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        filepath = os.path.join(filepath, 'expert.npy')
        self.actor.save(filepath)

    def load(self, filepath):
        self.actor.load(filepath)