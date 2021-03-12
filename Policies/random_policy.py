from Policies.base_policy import BasePolicy

import random
import gym

class RandomPolicy(BasePolicy):

    def __init__(self, ac_space, **kwargs):
       super().__init__(**kwargs)
       print(ac_space.shape)
       self.ac_space = ac_space
       self.is_discrete = isinstance(ac_space, gym.spaces.Discrete)
       self.ac_dim = ac_space.n if self.is_discrete else ac_space.shape[0]

    def build_graph(self):
        raise NotImplementedError

    def get_action(self, obs):
        if self.is_discrete:
            ac = [random.randint(0, self.ac_dim-1)] 
        else:
            ac = [self.ac_space.sample()]

        return ac

    def update(self, obs, acs):
        raise NotImplementedError

    def save(self, filepath):
    	raise NotImplementedError

    def restore(self, filepath):
    	raise NotImplementedError