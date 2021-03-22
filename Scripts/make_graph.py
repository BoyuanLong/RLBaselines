import os, sys

from numpy.core.fromnumeric import argmax
sys.path.append(os.path.join(sys.path[0], '..'))

import numpy as np
import matplotlib.pyplot as plt

def get_actions(agent):
    return np.array([np.argmax(i) for i in agent])

def action_diff(agent1, agent2):
    ac1 = get_actions(agent1)
    ac2 = get_actions(agent2)
    assert len(ac1) == len(ac2)
    diff = np.sum(ac1 != ac2) / len(ac1)
    print(diff)

def plot_q(q_table, desc):
    v_table = np.array([max(i) for i in q_table])
    v_table = v_table.reshape(8,8)
    fix, ax = plt.subplots()
    im = ax.imshow(v_table)
    ax.set_xticks(np.arange(v_table.shape[1]))
    ax.set_yticks(np.arange(v_table.shape[0]))

    for i in range(8):
        for j in range(8):
            text = ax.text(j, i, "{:.2f}".format(v_table[i, j]),
                        ha="center", va="center", color="white")

    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    # plt.show()
    plt.savefig('./q_table_{}.png'.format(desc))

def main():
    offline_agent_path = os.path.join('Experts', 'Offline', 'QLearning', 'expert.npy')
    online_agent_path = os.path.join('Experts', 'QLearning', 'expert.npy')
    offline_agent = np.load(offline_agent_path)
    online_agent = np.load(online_agent_path)
    plot_q(offline_agent, 'offline')
    plot_q(online_agent, 'online')
    # print(offline_agent)
    # x = np.arange(len(offline_agent.flatten()))
    # width = 0.2
    # plt.bar(x - width/2, offline_agent.flatten(), width, label='offline')
    # print(online_agent)
    # plt.bar(x + width/2, online_agent.flatten(), width, label='online')
    # plt.legend()
    # plt.savefig('./qvalue.png')
    
    # action_diff(offline_agent, online_agent)



if __name__ == '__main__':
    main()