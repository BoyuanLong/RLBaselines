from typing import Union
import gym
import numpy as np


def sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array')):
    ob = env.reset()

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
        acs.append(ac)
        ob, r, done, _ = env.step(ac)

        steps += 1
        next_obs.append(ob)
        rewards.append(r)

        rollout_done = done or (steps >= max_path_length)
        terminals.append(rollout_done)

        if rollout_done:
            break
    
    return obs, acs, rewards, next_obs, terminals, image_obs


def get_space_dim(space):
    if isinstance(space, gym.spaces.Discrete):
        dim = space.n
    else:
        dim = space.shape[0]
    return dim

def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.int),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.int),
            "action" : np.array(acs, dtype=np.int),
            "next_observation": np.array(next_obs, dtype=np.int),
            "terminal": np.array(terminals, dtype=np.float32)}