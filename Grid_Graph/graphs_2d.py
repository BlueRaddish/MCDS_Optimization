import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import random
import time
from tqdm import tqdm

# --- Graph Definitions ---
def generate_2D_grid_graph(rows, cols):
    G = nx.grid_2d_graph(rows, cols)
    return G

# --- Visualization Tools ---
def plot_graph(graph, pos=None):
    plt.figure(figsize=(6,6))
    pos = {(x,y):(y,-x) for x,y in graph.nodes()}
    nx.draw(graph, pos=pos, 
            node_color='lightgreen', 
            with_labels=True,
            node_size=600)

def overlay_subset(G, nodes):
    print(nodes)
    pos = {(x,y):(y,-x) for x,y in G.nodes()}
    plt.figure(figsize=(6,6))
    nx.draw(G, pos=pos, 
            node_color='lightgreen', 
            with_labels=True,
            node_size=600)
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color='orange')
    plt.show()

# --- Adjacency Logic ---
def adjacent_8(G, node):
    x, y = node
    adj = [(x-1, y-1),  (x-1, y),   (x-1, y+1),
           (x, y-1),                (x, y+1),
           (x+1, y-1),  (x+1, y),   (x+1, y+1)]
    return [n for n in adj if n in G.nodes()]

def adjacent_4(G, node):
    x, y = node
    adj = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    return [n for n in adj if n in G.nodes()]

def adjacent_graph(G, node):
    return list(G.neighbors(node))

# --- Subgraph and Set Properties ---
def is_connected_subgraph(G, nodes=None):
    if type(G) == 'networkx.classes.graph.Graph':
        subG = G.subgraph(nodes)
        return nx.is_connected(subG)
    return "undefined function for this type of graph"

def is_dominating_set(G, nodes):
    covered = set()
    for cell in nodes:
        covered.update(adjacent_8(G, cell))
    return len(covered) == G.number_of_nodes()

def is_connected_dominating_set(G, nodes):
    visited = set()
    stack = [next(iter(nodes))]
    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            for neighbor in adjacent_4(G, current):
                if neighbor in nodes and neighbor not in visited:
                    stack.append(neighbor)
    if len(visited) != len(nodes):
        return False
    return True and is_dominating_set(G, nodes)

# --- Node Selection Strategies ---
def select_starting_node_8way_probablistic(G):
    degrees = [G.degree(n) for n in G.nodes()]
    total = sum(degrees)
    probabilities = [deg / total for deg in degrees]
    nodes = list(G.nodes())
    selected_node = random.choices(nodes, weights=probabilities, k=1)[0]
    return selected_node

def select_starting_node_max_neighbors(G):
    max_degree = max(G.degree(n) for n in G.nodes())
    candidates = [n for n in G.nodes() if G.degree(n) == max_degree]
    return random.choice(candidates)

# --- Utility Functions ---
def measure_runtime(func, *args, trials=1, **kwargs):
    start_time = time.time()
    results = []
    for _ in tqdm(range(trials), desc="Running trials"):
        result = func(*args, **kwargs)
        results.append(result)
    end_time = time.time()
    elapsed_time = (end_time - start_time) / trials
    return results, elapsed_time

def get_minimum_from_trials(select, dset, G, trials=100):
    min_len = float('inf')
    min_set = None
    for _ in range(trials):
        dom_set = dset(G, select(G))
        if len(dom_set) < min_len:
            min_len = len(dom_set)
            min_set = dom_set
    return min_set, min_len
