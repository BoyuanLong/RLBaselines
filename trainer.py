import time
import gym

from Policies.random_policy import RandomPolicy

class Trainer(object):

    def __init__(self, name):
        seed = 0

        # Make the gym environment
        self.env = gym.make(name)
        self.env.seed(seed)
        self.policy = RandomPolicy(self.env.action_space)


    def sample_trajectory(self, render=False, render_mode=('rgb_array')):
        env = self.env
        ob = env.reset()
        policy = self.policy

        obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
        steps = 0
        while True:
            if render:
                if 'rgb_array' in render_mode:
                    if hasattr(env, 'sim'):
                        image_obs.append(env.sim.render(camera_name='track', height=500, width=500)[::-1])
                    else:
                        image_obs.append(env.render(mode=render_mode))
                if 'human' in render_mode:
                    env.render(mode=render_mode)
            
            obs.append(ob)
            ac = policy.get_action(ob)

            ac = ac[0]
            acs.append(ac)

            ob, r, done, _ = env.step(ac)

            steps += 1
            next_obs.append(ob)
            rewards.append(r)

            rollout_done = done
            terminals.append(rollout_done)

            if rollout_done:
                print('finished')
                break
        
        return obs, acs, rewards, next_obs, terminals, image_obs
        