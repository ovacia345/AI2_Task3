import numpy as np

def min_children(network, evidence, query):
    nr_children = {node: 0 for node in network.nodes}
    for node in network.nodes:
        for parent in network.parents[node]:
            nr_children[parent] += 1

    return get_elim_order(nr_children, evidence, query)

def min_parents(network, evidence, query):
    nr_parents = {node: 0 for node in network.nodes}
    for node in network.nodes:
        nr_parents[node] += len(network.parents[node])

    return get_elim_order(nr_parents, evidence, query)

def min_neighbours(network, evidence, query):
    nr_neighbours = {node: 0 for node in network.nodes}
    for node in network.nodes:
        for parent in network.parents[node]:
            nr_neighbours[node] += 1
            nr_neighbours[parent] += 1

    return get_elim_order(nr_neighbours, evidence, query)

def min_weight(network, evidence, query):
    weights = {node: 1 for node in network.nodes}
    for node in network.nodes:
        for parent in network.parents[node]:
            weights[node] *= len(network.values[parent])
            weights[parent] *= len(network.values[node])

    return get_elim_order(weights, evidence, query)

def random(network, evidence, query):
    weights = {node: np.random.rand() for node in network.nodes}

    return get_elim_order(weights, evidence, query)

def get_elim_order(weights, evidence, query):
    if query not in weights.keys() or any(
            e not in weights.keys() for e in evidence.keys()):
        raise ValueError("Invalid key encountered")

    for key in evidence.keys():
        weights.pop(key)
    weights.pop(query)

    indices = np.argsort(weights.values())
    elim_order = np.array(weights.keys())[indices]
    return elim_order
