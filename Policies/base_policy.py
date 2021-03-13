class BasePolicy(object):

    def __init__(self, **kwargs):
       super(BasePolicy, self).__init__(**kwargs)

    def get_action(self, obs):
        raise NotImplementedError

    def update(self, obs, acs):
        raise NotImplementedError

    def save(self, filepath):
    	raise NotImplementedError

    def load(self, filepath):
    	raise NotImplementedError