import numpy as np
from collections import defaultdict

def min_neighbours(network, evidence, query):
    nr_neighbours = {node: 0 for node in network.nodes}
    for node in network.nodes:
        for parent in network.parents[node]:
            nr_neighbours[node] += 1
            nr_neighbours[parent] += 1

    for key in evidence.keys():
        nr_neighbours.pop(key)
    nr_neighbours.pop(query)
    indices = np.argsort(nr_neighbours.values())
    elim_order = np.array(nr_neighbours.keys())[indices]
    return elim_order

def min_weight(network, evidence, query):
    neighbours_weights = {node: [] for node in network.nodes}
    for node in network.nodes:
        for parent in network.parents[node]:
            neighbours_weights[node] += [len(network.values[parent])]
            neighbours_weights[parent] += [len(network.values[node])]

    for key in evidence.keys():
        neighbours_weights.pop(key)
    neighbours_weights.pop(query)
    neighbours_weights_products = [np.prod(neighbours_weights[node]) for node in neighbours_weights.keys()]
    indices = np.argsort(neighbours_weights_products)
    elim_order = np.array(neighbours_weights.keys())[indices]
    return elim_order