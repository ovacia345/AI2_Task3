import numpy as np

def min_neighbours(network, evidence, query):
    neighbours = {node: set() for node in network.nodes}
    for node in network.nodes:
        neighbours[node].update(network.parents[node])
        for parent in network.parents[node]:
            neighbours[parent].add(node)

    for key in evidence.keys():
        neighbours.pop(key)
    neighbours.pop(query)
    nr_neighbours = [len(neighbours[node]) for node in neighbours.keys()]
    indices = np.argsort(nr_neighbours)
    elim_order = np.array(neighbours.keys())[indices]
    return elim_order
