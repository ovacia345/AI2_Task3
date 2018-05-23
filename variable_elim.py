"""
@Author: Joris van Vugt, Moira Berens

Implementation of the variable elimination algorithm for AISPAML assignment 3

"""
import numpy as np
from factor import Factor
import datetime
from heuristics import *

class VariableElimination():

    def __init__(self, network):
        self.network = network
        self.network.values = {node: sorted(self.network.values[node]) for node in self.network.values}


    def run(self, query, observed, elim_order):
        self.start_time = datetime.datetime.now()

        factors = np.array([])
        for node in self.network.nodes:
            factor = self._makeFactor(node, observed)
            if factor.nr_nodes > 0:
                factors = np.append(factors, factor)

        for node in elim_order:
            node_factors_indices = [i for i, factor in enumerate(factors) if node in factor.nodes]
            node_factors = factors[node_factors_indices]

            product = Factor.product(node_factors)

            if product.nr_nodes > 1:
                marginalization = product.marginalize(node)
                factors = np.append(factors, marginalization)

            factors = np.delete(factors, node_factors_indices)


        result = Factor.product(factors)

        return result.normalize(query)

    def _makeFactor(self, node, observed):
        nodes = np.array([node] + self.network.parents[node])
        probs = self.network.probabilities[node].values

        nodes_in_observed = nodes[np.in1d(nodes, observed.keys())]
        if nodes_in_observed.size > 0:
            if nodes.size > nodes_in_observed.size:
                factor = Factor(nodes, probs, self.network)
                return factor.reduce(nodes_in_observed, observed)
            else:
                return Factor()
        else:
            return Factor(nodes, probs, self.network)
