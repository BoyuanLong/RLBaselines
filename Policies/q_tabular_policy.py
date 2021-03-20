from Policies.base_policy import BasePolicy
import numpy as np
import random

class QPolicy(BasePolicy):

    def __init__(self, args, **kwargs):
        super(QPolicy, self).__init__(**kwargs)
        self.lr = args.learning_rate
        self.gamma = args.gamma
        self.ob_dim = args.ob_dim
        self.ac_dim = args.ac_dim
        self.q_table = np.zeros(self.ob_dim * self.ac_dim).reshape(self.ob_dim, self.ac_dim).astype(np.float32)

        self.is_training = False
        self.e_greedy = args.e_greedy
        self.epsilon = 0.0
        self.e_decay_rate = args.e_decay_rate

    def training(self):
        self.is_training = True
        if self.e_greedy:
            self.epsilon = 1.0
    
    def testing(self):
        self.is_training = False
        self.epsilon = 0.0
    
    def get_action(self, obs):
        if self.is_training and random.random() < self.epsilon:
            ac = random.randint(0, self.ac_dim-1)
        else:
            ac = np.argmax(self.q_table[obs])

        return ac

    def update(self, obs, acs, rews, next_obs, terminals):
        for i in range(len(obs)):
            q_next = np.max(self.q_table[next_obs[i]])
            self.q_table[obs[i]][acs[i]] = (1 - self.lr) * self.q_table[obs[i]][acs[i]] \
                + self.lr * (rews[i] + self.gamma * q_next)
        
        self.epsilon = max(self.epsilon - self.e_decay_rate, 0.0)

        return np.sum(rews)

    def save(self, filepath):
        np.save(filepath, self.q_table)
        print(self.q_table)
        print("Q Table has been saved to file: {}.".format(filepath))

    def load(self, filepath):
        self.q_table = np.load(filepath)
        print("Q Table has been loaded from file: {}".format(filepath))
        print(self.q_table)