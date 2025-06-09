import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import random
import time
from tqdm import tqdm

# --- Graph Definitions ---
def generate_2D_grid_graph(rows, cols):
    """Generate a 2D grid graph with the given number of rows and columns."""
    G = nx.grid_2d_graph(rows, cols)
    return G

def create_hole_at_location(G, location=None):
    """
    Remove a node from the 2D grid graph at the specified location or at a random location.
    Returns the modified graph and the location of the hole.
    """
    all_nodes = list(G.nodes())
    if not all_nodes:
        raise ValueError("Graph has no nodes to remove.")
    if location is not None:
        if location not in G.nodes():
            raise ValueError(f"Location {location} is not a valid node in the graph.")
        hole = location
    else:
        hole = random.choice(all_nodes)
    G.remove_node(hole)
    return G, hole

def generate_2D_grid_graph_with_holes(rows, cols, num_holes):
    """Generate a 2D grid graph and remove a specified number of random nodes (holes)."""
    G = nx.grid_2d_graph(rows, cols)
    all_nodes = list(G.nodes())
    if num_holes > len(all_nodes):
        raise ValueError("Number of holes exceeds number of nodes in the grid.")
    holes = random.sample(all_nodes, num_holes)
    G.remove_nodes_from(holes)
    return G, holes

def generate_random_graph_with_avg_degree(nodes, neighbors):
    """
    Generate a random undirected graph with a given number of nodes and average degree.
    Uses the Erdős–Rényi model.
    """
    if nodes < 2 or neighbors < 0 or neighbors > nodes-1:
        raise ValueError("Invalid parameters: m must be >= 2 and 0 <= n <= m-1.")
    p = neighbors / (nodes - 1)
    G = nx.erdos_renyi_graph(nodes, p)
    return G

# --- Visualization Tools ---
def plot_graph(graph, pos=None):
    """Plot a 2D grid graph using matplotlib."""
    plt.figure(figsize=(6,6))
    pos = {(x,y):(y,-x) for x,y in graph.nodes()}
    nx.draw(graph, pos=pos, 
            node_color='lightgreen', 
            with_labels=True,
            node_size=600)

def overlay_subset(G, nodes):
    """Plot a 2D grid graph and highlight a subset of nodes in orange."""
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
    """Return the list of 8-way adjacent nodes for a given node in the grid."""
    x, y = node
    adj = [(x-1, y-1),  (x-1, y),   (x-1, y+1),
           (x, y-1),                (x, y+1),
           (x+1, y-1),  (x+1, y),   (x+1, y+1)]
    return [n for n in adj if n in G.nodes()]

def adjacent_4(G, node):
    """Return the list of 4-way adjacent nodes for a given node in the grid."""
    x, y = node
    adj = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    return [n for n in adj if n in G.nodes()]

def adjacent_graph(G, node):
    """Return the list of neighbors for a given node in the graph."""
    return list(G.neighbors(node))

# --- Subgraph and Set Properties ---
def is_connected_subgraph(G, nodes=None):
    """Check if the subgraph induced by the given nodes is connected."""
    if type(G) == 'networkx.classes.graph.Graph':
        subG = G.subgraph(nodes)
        return nx.is_connected(subG)
    return "undefined function for this type of graph"

def is_dominating_set(G, nodes):
    """Check if the given set of nodes is a dominating set in the graph."""
    covered = set()
    for cell in nodes:
        covered.update(adjacent_8(G, cell))
    return len(covered) == G.number_of_nodes()

def is_connected_dominating_set(G, nodes):
    """Check if the given set of nodes is a connected dominating set in the graph."""
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
    """Select a starting node with probability proportional to its degree (8-way adjacency)."""
    degrees = [G.degree(n) for n in G.nodes()]
    total = sum(degrees)
    probabilities = [deg / total for deg in degrees]
    nodes = list(G.nodes())
    selected_node = random.choices(nodes, weights=probabilities, k=1)[0]
    return selected_node

def select_starting_node_max_neighbors(G):
    """Select a starting node with the maximum number of neighbors."""
    max_degree = max(G.degree(n) for n in G.nodes())
    candidates = [n for n in G.nodes() if G.degree(n) == max_degree]
    return random.choice(candidates)

# --- Utility Functions ---
def measure_runtime(func, *args, trials=1, **kwargs):
    """Measure the average runtime of a function over a number of trials."""
    start_time = time.time()
    results = []
    for _ in tqdm(range(trials), desc="Running trials"):
        result = func(*args, **kwargs)
        results.append(result)
    end_time = time.time()
    elapsed_time = (end_time - start_time) / trials
    return results, elapsed_time

def get_minimum_from_trials(select, dset, G, trials=100):
    """Run multiple trials to find the minimum dominating set found by a selection strategy."""
    min_len = float('inf')
    min_set = None
    for _ in range(trials):
        dom_set = dset(G, select(G))
        if len(dom_set) < min_len:
            min_len = len(dom_set)
            min_set = dom_set
    return min_set, min_len
