import numpy as np

def discounted_reward(gamma, rewards):
    '''
        Sum_{t=0}^{T-1} gamma^t * r_t
    '''
    indices = np.arange(0, len(rewards))
    discounts = np.power(gamma, indices)
    discounted_rewards = discounts * rewards
    sum = np.sum(discounted_rewards)
    list_of_sum = np.repeat(sum, len(rewards))
    return list_of_sum

def discounted_cumsum(gamma, rewards):
    '''
        Sum_{t=t'}^{T-1} gamma^(t-t') * r_{t} for each t'
    '''
    indices = np.arange(0, len(rewards))
    discounts = np.power(gamma, indices)
    
    list_of_sum = [np.sum(discounts[:len(rewards) - t] * rewards[t:]) for t in range(len(rewards))]

    return np.array(list_of_sum)

if __name__ == "__main__":
    rewards = [1,1,1,1,1,1,1,1]
    gamma = 0.1
    print(discounted_reward(gamma, rewards))
    print(discounted_cumsum(gamma, rewards))