import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import random

# Help functions

# Plots a 2D grid graph with labeled nodes for visualization.
def plot_graph(graph, pos=None):
    plt.figure(figsize=(6,6))
    pos = {(x,y):(y,-x) for x,y in graph.nodes()}
    nx.draw(graph, pos=pos, 
            node_color='lightgreen', 
            with_labels=True,
            node_size=600)
    
# Checks if the subgraph induced by a set of nodes is a connected subgraph.
def is_connected_subgraph(G, nodes=None):
    if type(G) == 'networkx.classes.graph.Graph':
        subG = G.subgraph(nodes)
        return nx.is_connected(subG)
    return "undefined function for this type of graph"

# Returns the 8-way (including diagonals) adjacent nodes of a given node in the graph.
def adjacent_8(G, node):
    x, y = node
    adj = [(x-1, y-1),  (x-1, y),   (x-1, y+1),
           (x, y-1),                (x, y+1),
           (x+1, y-1),  (x+1, y),   (x+1, y+1)]
    
    # Filter out nodes that are not in the graph
    return [n for n in adj if n in G.nodes()]

# Returns the 4-way (no diagonals) adjacent nodes of a given node in the graph.
def adjacent_4(G, node):
    x, y = node
    adj = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    
    # Filter out nodes that are not in the graph
    return [n for n in adj if n in G.nodes()]

# Checks if a set of nodes forms a dominating set (covers all nodes) using 8-way adjacency.
def is_dominating_set(G, nodes):
    covered = set()
    for cell in nodes:
        covered.update(adjacent_8(G, cell))
    return len(covered) == G.number_of_nodes()

# Checks if a set of nodes is a connected dominating set using 4-way adjacency.
def is_connected_dominating_set(G, nodes): # Needs to be simplified - EDIT LATR
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

# Generates a 2D grid graph with the specified number of rows and columns.
def generate_2D_grid_graph(rows, cols):
    G = nx.grid_2d_graph(rows, cols)
    return G

# Selects a starting node probabilistically, weighted by node degree (8-way adjacency).
def select_starting_node_8way_probablistic(G):
    degrees = [G.degree(n) for n in G.nodes()]
    total = sum(degrees)
    probabilities = [deg / total for deg in degrees]
    nodes = list(G.nodes())
    selected_node = random.choices(nodes, weights=probabilities, k=1)[0]
    return selected_node

# Selects a starting node probabilistically from the nodes with highest degree.
def select_starting_node_max_neighbors(G):
    max_degree = max(G.degree(n) for n in G.nodes())
    candidates = [n for n in G.nodes() if G.degree(n) == max_degree]
    return random.choice(candidates)

# Overlays a subset of nodes on the graph for visualization.
def overlay_subset(G, nodes):
    print(nodes)
    pos = {(x,y):(y,-x) for x,y in G.nodes()}
    plt.figure(figsize=(6,6))
    nx.draw(G, pos=pos, 
            node_color='lightgreen', 
            with_labels=True,
            node_size=600)
    
    # Highlight the dominating set
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color='orange')
    
    plt.show()