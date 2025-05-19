import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools

# Help functions
def plot_graph(graph, pos=None):
    plt.figure(figsize=(6,6))
    pos = {(x,y):(y,-x) for x,y in graph.nodes()}
    nx.draw(graph, pos=pos, 
            node_color='lightgreen', 
            with_labels=True,
            node_size=600)
    
def is_connected_subgraph(G, nodes=None):
    if type(G) == 'networkx.classes.graph.Graph':
        subG = G.subgraph(nodes)
        return nx.is_connected(subG)
    return "undefined function for this type of graph"

def adjacent_8(G, node):
    """
    Returns the 8 adjacent nodes of a given node in a graph.
    """
    x, y = node
    adj = [(x-1, y-1),  (x-1, y),   (x-1, y+1),
           (x, y-1),                (x, y+1),
           (x+1, y-1),  (x+1, y),   (x+1, y+1)]
    
    # Filter out nodes that are not in the graph
    return [n for n in adj if n in G.nodes()]

def adjacent_4(G, node):
    """
    Returns the 4 adjacent nodes of a given node in a graph.
    """
    x, y = node
    adj = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    
    # Filter out nodes that are not in the graph
    return [n for n in adj if n in G.nodes()]

def is_dominating_set(G, nodes):
    covered = set()
    for cell in nodes:
        covered.update(adjacent_8(G, cell))
    return len(covered) == G.number_of_nodes()

def is_connected_dominating_set(G, nodes):
    """
    Check if a set of nodes is a connected dominating set in the graph.
    """
    # Check if all nodes are adjacently connected using adjacent_4
    # EDIT LATR
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



def generate_2D_grid_graph(rows, cols):
    """
    Generate a 2D grid graph with the specified number of rows and columns.
    """
    G = nx.grid_2d_graph(rows, cols)
    return G