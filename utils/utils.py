from typing import Union



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