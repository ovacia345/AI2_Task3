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


    def run(self, query, observed, elim_order, log):
        self.wall_clock_time = datetime.datetime.now()

        factors = np.array([])
        for node in self.network.nodes:
            factor = self._makeFactor(node, observed, log)
            if factor.nr_nodes > 0:
                factors = np.append(factors, factor)

                log.write("factor " + str(factors.size) + ":\n" + str(factor) + "\n")

        for node in elim_order:
            node_factors_indices = [i for i, factor in enumerate(factors) if node in factor.nodes]
            node_factors = factors[node_factors_indices]

            product, nr_multiplication_steps = Factor.product(node_factors.copy())

            self.nr_multiplication_steps += nr_multiplication_steps

            if product.nr_nodes > 1:
                marginalization = product.marginalize(node)
                factors = np.append(factors, marginalization)

                log.write("\n---------------------------------------------------------------\n"
                          "Eliminating " + node + "\n\n")
                log.write("Multiply\n")
                for i, node_factor in enumerate(node_factors):
                    log.write("factor " + str(i + 1) + ":\n" + str(node_factor) + "\n")
                log.write("\nProduct:\n" + str(product) + "\n")
                log.write("\nMarginalization:\n" + str(marginalization))

            factors = np.delete(factors, node_factors_indices)


        result_product, nr_multiplication_steps = Factor.product(factors.copy())
        result = result_product.normalize(query)

        log.write("\n\n---------------------------------------------------------------\n"
                  "Producing query " + query + "\n\n")
        log.write("Multiply\n")
        for i, factor in enumerate(factors):
            log.write("factor " + str(i + 1) + ":\n" + str(factor) + "\n")
        log.write("\nProduct:\n" + str(result_product) + "\n")
        log.write("\nNormalization:\n" + str(result))

        self.wall_clock_time = datetime.datetime.now() - self.wall_clock_time
        self.nr_multiplication_steps += nr_multiplication_steps

        return result

    def _makeFactor(self, node, observed, log):
        nodes = np.array([node] + self.network.parents[node])
        probs = self.network.probabilities[node].values

        nodes_in_observed = nodes[np.in1d(nodes, observed.keys())]
        if nodes_in_observed.size > 0:
            if nodes.size > nodes_in_observed.size:
                factor = Factor(nodes, probs, self.network)
                factor_reduced = factor.reduce(nodes_in_observed, observed)

                log.write("\nReduce\n")
                log.write(str(factor))
                log.write("Evidence: " + str(observed) + "\n\n")

                return factor_reduced
            else:
                return Factor()
        else:
            return Factor(nodes, probs, self.network)
