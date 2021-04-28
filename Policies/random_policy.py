from Policies.base_policy import BasePolicy

import random
import gym

class RandomPolicy(BasePolicy):

    def __init__(self, ac_space, **kwargs):
       super().__init__(**kwargs)
       self.is_discrete = True
       self.ac_dim = ac_space

    def get_action(self, obs):
        if self.is_discrete:
            ac = random.randint(0, self.ac_dim-1)
        else:
            ac = self.ac_space.sample()
        return ac

    def update(self, obs, acs):
        raise NotImplementedError

    def save(self, filepath):
    	raise NotImplementedError

    def load(self, filepath):
    	raise NotImplementedError