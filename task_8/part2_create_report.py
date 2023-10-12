import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tqdm import tqdm
from fa2 import ForceAtlas2

df = pd.read_csv('roberta_metrics.csv', index_col='Model')

basic = np.array(df.drop('TheO*', axis=1))
self_res = np.diagonal(basic)
basic = basic / self_res
np.fill_diagonal(basic, 0)

onion = 2 * df['TheO*']

shape_final = basic.shape[0] + 1
g_final = np.zeros((shape_final, shape_final))

g_final[:-1, :-1] = basic
g_final[-1, :-1] = onion
g_final = g_final + g_final.T

G = nx.complete_graph(shape_final)
forceatlas2 = ForceAtlas2()
original_positions = np.array(list(nx.circular_layout(G).values()))
positions = forceatlas2.forceatlas2(g_final, pos=original_positions, iterations=1000000)


node_sizes = [0.5] * 11
for i in [1, 2, 7]:
    node_sizes[i] = 0.25
for i in [3, 8]:
    node_sizes[i] = 1.0 

label_positions = []
for i in range(len(positions)):
    a, b = positions[i]
    if i in [1, 2, 7]:
        label_positions.append((a, b + 0.75))
    elif i in [3, 8]:
        label_positions.append((a, b + 1.5))
    else:
        label_positions.append((a, b + 1.0))

topics = [['Web'], ['Web'], ['Web'], ['Web', 'Re'], ['Re'], ['Twi', 'Re'], 
          ['Editing'], ['Oni'], ['Re'], ['Web', 'Re', 'Oni'], ['Oni']]

color_map = {'Web': 'gray', 'Re': 'red', 'Oni': 'green', 'Twi': 'blue', 'Editing': 'orange'}

labels = list(df.columns)
labels = {i: labels[i] for i in range(len(labels))}

fig = plt.figure(figsize=(10, 10))
nx.draw_networkx_edges(G, positions, edge_color="gray", alpha=0.5)

for i in range(len(G.nodes)):
    attributes = topics[i]
    colors = [color_map[topic] for topic in attributes]
    plt.pie([1] * len(attributes), center=positions[i], colors=colors, radius=node_sizes[i])

nx.draw_networkx_labels(G, label_positions, labels, 
    font_size=16, font_color="black", font_family='times new roman', font_weight='bold')

plt.ylim(-15,15)
plt.xlim(-25,25)

handles_, labels_ = [], []
for i in color_map:
    empty_patch = mpatches.Patch(color=color_map[i], label=i)
    handles_.append(empty_patch)
    labels_.append(i)

plt.legend(handles_, labels_)
plt.savefig('./reports/forceatlas.png')