import numpy as np

def chooseHeuristic(network, evidence, query, heuristic):
    if heuristic == 'least-incoming-arcs first':
        return least_incoming_arcs_first(network, evidence, query)

def least_incoming_arcs_first(network, evidence, query):
    nodes = np.array(network.nodes)
    nodes = nodes[np.in1d(nodes, evidence.keys(), invert = True)]
    nodes = nodes[nodes != query]

    nr_incoming_arcs = [len(network.parents[node]) for node in nodes]
    sorted_indices = np.argsort(nr_incoming_arcs)

    return nodes[sorted_indices]
