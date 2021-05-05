import numpy as np
from matplotlib import pyplot as plt


def val_iter(rewards,t_matrix,theta,gamma):

    delta = theta * 10
    deltas = []

    n_state = np.shape(rewards)[0]
    n_action = np.shape(rewards)[1]

    val_func = np.zeros(n_state)

    while delta>theta:
        delta = 0
        for i in range(n_state):
            v_temp = val_func[i]
            max_buffer = np.zeros(n_action)
            for j in range(n_action):
                max_buffer[j] = rewards[i][j] + gamma*np.dot(val_func,t_matrix[i][j])
            val_func[i] = np.max(max_buffer)
            delta = np.max((delta,np.abs(v_temp-val_func[i])))
        deltas.append(delta)
    
    return val_func,deltas

def policy_iter(rewards,t_matrix,theta,gamma):

    n_state = np.shape(rewards)[0]
    n_action = np.shape(rewards)[1]

    val_func = np.zeros(n_state)
    policy = np.zeros(n_state)

    val_old = np.zeros(np.shape(val_func))

    deltas = []
    delta = theta*10

    theta_val = 1e-10

    while delta>theta:
    #policy evaluation
        delta_val = theta_val*10
        while delta_val>theta_val:
            delta = 0
            for i in range(n_state):
                v_temp = val_func[i]
                action = int(policy[i])
                val_func[i] =rewards[i][action] + gamma*np.dot(val_func,t_matrix[i][action])
                delta_val = np.max((delta,np.abs(v_temp-val_func[i])))
    
        q_next = get_q_func(rewards=rewards,t_matrix=t_matrix,val_func=val_func,gamma=gamma)
        policy = get_policy(q_next)

        delta = np.max(np.abs(val_old-val_func))
        deltas.append(delta)
        val_old = np.copy(val_func)
    
    return val_func,policy,deltas
    
def get_q_func(rewards,t_matrix,val_func,gamma):

    n_state = np.shape(rewards)[0]
    n_action = np.shape(rewards)[1]

    q_func = np.zeros((n_state,n_action))

    for i in range(n_state):
        max_buffer = np.zeros(n_action)
        for j in range(n_action):
            max_buffer[j] = rewards[i][j] + gamma*np.dot(val_func,t_matrix[i][j])
        q_func[i] = max_buffer
        #expert_policy[i] = np.argmax(max_buffer)
    return q_func

def get_policy(q_func):

    n_state = np.shape(q_func)[0]
    policy = np.zeros(n_state)

    for i in range(n_state):
        policy[i] = np.argmax(q_func[i])

    return policy

def offline_val_iter(rewards,theta,gamma,dataset,n_next_state,max_iter=1000):

    delta = theta * 10
    deltas = []

    exp_scale = 1/n_next_state

    n_state = np.shape(rewards)[0]
    n_action = np.shape(rewards)[1]

    val_func = np.zeros(n_state)

    deltas = []
    #for iter in range(0,100):
    iters = 0
    while delta>theta and iters<max_iter:
        delta = 0
        for i in range(n_state):
            v_temp = val_func[i]
            max_buffer = np.zeros(n_action)
            for j in range(n_action):
                next_state_samples = np.random.choice(dataset[str(i)][str(j)],size=n_next_state)
                max_buffer[j] = rewards[i][j] + gamma*exp_scale*np.sum(np.take(val_func,next_state_samples))
            val_func[i] = np.max(max_buffer)
            delta = np.max((delta,np.abs(v_temp-val_func[i])))
        deltas.append(delta)
        iters+=1
    
    return val_func,deltas

def offline_policy_iter(rewards,t_matrix,theta,gamma,dataset,n_next_state,n_episodes,T,max_iter=10):

    n_state = np.shape(rewards)[0]
    n_action = np.shape(rewards)[1]

    exp_scale = 1/n_next_state
    episode_scale = 1/n_episodes

    val_func = np.zeros(n_state)
    policy = np.zeros(n_state)

    deltas = []
    delta = theta*10

    policy_stable = False

    iter = 0
    while delta>theta and iter<max_iter and not policy_stable:
        print('Iteration '+str(iter))
        val_next = np.zeros(n_state)
        #print(val_func)
        for i in range(n_state):
            for q in range(n_episodes):
                #reward_steps = np.zeros(T)
                state = i
                for step in range(T):
                    action = int(policy[state])
                    val_next[i] += episode_scale*(gamma**step)*rewards[state][action]
                    #reward_steps[step] = episode_scale*(gamma**step)*rewards[state][action]
                    #print('Value: '+str(val_next[i]))
                    state = np.random.choice(dataset[str(state)][str(action)])
                    #state = np.random.choice(n_state,p=t_matrix[state][action])
                #lt.plot(reward_steps)
                #plt.savefig('Reward Steps - State '+str(i)+' Episode '+str(q)+'.png')
                #plt.close()

        plt.plot(val_next)
        plt.title('Val Next: '+str(iter))
        plt.savefig('Val Next'+str(iter)+'.png')
        plt.close()

        old_policy = np.copy(policy)
        #print('Pre Update')
        #print(old_policy)
            
        for i in range(n_state):
            max_buffer = np.zeros(n_action)
            for j in range(n_action):
                next_state_samples = np.random.choice(dataset[str(i)][str(j)],size=n_next_state)
                max_buffer[j] = rewards[i][j] + gamma*exp_scale*np.sum(np.take(val_next,next_state_samples))
            policy[i] = np.argmax(max_buffer)
        #print('Post Update')
        #print(old_policy)

        policy_stable = np.array_equal(old_policy,policy)
        print('Policy Stable: '+str(policy_stable))

        #q_next = get_q_func(rewards=rewards,t_matrix=t_matrix,val_func=val_next,gamma=gamma)
        #policy = get_policy(q_next)
        
        plt.plot(policy)
        plt.title('Policy: '+str(iter))
        plt.savefig('Policy '+str(iter)+'.png')
        plt.close()

        #print(val_func)
        delta = np.max(np.abs(val_next-val_func))
        print('Delta: '+str(delta))
        deltas.append(delta)
        val_func = np.copy(val_next)
        iter+=1
    
    return val_func,policy,deltas