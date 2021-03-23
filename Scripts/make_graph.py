import os, sys

from numpy.core.fromnumeric import argmax
sys.path.append(os.path.join(sys.path[0], '..'))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

ACTIONS = ['L', 'D', 'R', 'U']

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def get_actions(agent):
    return np.array([np.argmax(i) for i in agent])

def plot_actions(action, desc):
    dim = int(np.sqrt(len(action)))
    assert dim ** 2 == len(action)
    action = action.reshape((dim, dim))

    fig, ax = plt.subplots()
    im, cbar = heatmap(action, np.arange(dim), np.arange(dim), ax=ax, cmap="Wistia")

    ax.set_title("{}x{} Policy Graph ({} Agent)".format(dim, dim, desc))

    for i in range(dim):
        for j in range(dim):
            text = ax.text(j, i, "{}".format(ACTIONS[action[i, j]]),
                        ha="center", va="center", color="black")

    plt.savefig('./action_{}.png'.format(desc))

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
    im, cbar = heatmap(v_table, np.arange(8), np.arange(8), ax=ax, cmap="Wistia")

    texts = annotate_heatmap(im, v_table)

    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    # plt.show()
    ax.set_title("Value function ({} Agent)".format(desc))
    plt.savefig('./q_table_{}.png'.format(desc))

def main():
    offline_agent_path = os.path.join('Experts', 'Offline', 'QLearning', 'expert.npy')
    online_agent_path = os.path.join('Experts', 'QLearning', 'expert.npy')
    offline_agent = np.load(offline_agent_path)
    online_agent = np.load(online_agent_path)
    online_action = get_actions(online_agent)
    offline_action = get_actions(offline_agent)
    plot_actions(online_action, "online")
    plot_actions(offline_action, "offline")
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