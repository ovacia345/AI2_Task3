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

        self.wall_clock_time = 0

        self.nr_multiplication_steps = 0


    def run(self, query, observed, elim_order):
        self.wall_clock_time = datetime.datetime.now()

        factors = np.array([])
        for node in self.network.nodes:
            factor = self._makeFactor(node, observed)
            if factor.nr_nodes > 0:
                factors = np.append(factors, factor)

        for node in elim_order:
            node_factors_indices = [i for i, factor in enumerate(factors) if node in factor.nodes]
            node_factors = factors[node_factors_indices]

            product, nr_multiplication_steps = Factor.product(node_factors)

            self.nr_multiplication_steps += nr_multiplication_steps

            if product.nr_nodes > 1:
                marginalization = product.marginalize(node)
                factors = np.append(factors, marginalization)

            factors = np.delete(factors, node_factors_indices)


        result, nr_multiplication_steps = Factor.product(factors)
        result = result.normalize(query)

        self.wall_clock_time = datetime.datetime.now() - self.wall_clock_time
        self.nr_multiplication_steps += nr_multiplication_steps

        return result

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
