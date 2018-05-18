"""
@Author: Joris van Vugt, Moira Berens

Implementation of the variable elimination algorithm for AISPAML assignment 3

"""
import numpy as np

class VariableElimination():

    def __init__(self, network):
        self.network = network
        self.network.values = {node: sorted(self.network.values[node]) for node in self.network.values}


    def run(self, query, observed, elim_order):
        factors = np.array([])
        for node in self.network.nodes:
            factor = self.makeFactor(node, observed)
            if factor.nr_nodes > 0:
                factors = np.append(factors, factor)

        for node in elim_order:
            node_factors_indices = [i for i, factor in enumerate(factors) if node in factor.nodes]

            product = Factor.product(factors[node_factors_indices])

            marginalization = product.marginalize(node)

            factors = np.append(factors, marginalization)
            factors = np.delete(factors, node_factors_indices)


        result = Factor.product(factors)

        return result.normalize()

    def makeFactor(self, node, observed):
        nodes = np.array([node] + self.network.parents[node])
        probs = self.network.probabilities[node].values

        nodes_in_observed = nodes[np.flatnonzero(np.in1d(nodes, observed.keys()))]
        if nodes_in_observed.size > 0:
            if nodes.size > nodes_in_observed.size:
                factor = Factor(nodes, probs, self.network)
                return factor.reduce(observed, nodes_in_observed)
            else:
                return Factor()
        else:
            return Factor(nodes, probs, self.network)



class Factor():

    def __init__(self, nodes = np.array([]), probs = np.array([]), network = None):
        self.nodes = nodes
        self.nr_nodes = self.nodes.size

        self.probs = probs

        self.network = network


    @staticmethod
    def product(factors):
        for i in xrange(factors.size - 1):
            factors[1] = factors[0].times(factors[1])
            factors = np.delete(factors, 0)
        return factors[0]

    @staticmethod
    def get_probs(factor1, factor2, prob_column, nr_common_nodes, nodes):
        values1 = factor1.probs[:, :-1]  # Includes common_nodes columns
        values2 = factor2.probs[:, nr_common_nodes:-1]  # Excludes common_nodes columns

        if values1[:, nr_common_nodes:].size != 0 and values2.size != 0:
            nr_rows = np.prod([len(factor1.network.values[node] for node in nodes)])

            probs = [np.repeat(values1, nr_rows / len(values1), axis=0)]
            probs += [np.tile(values2.T, nr_rows / len(values2)).T]
        elif values2.size != 0:
            probs = [np.repeat(values1, len(values2) / len(values1), axis=0)]
            probs += [values2]
        else:
            probs = [values1]
        probs += [prob_column]

        return np.hstack(probs)


    def times(self, factor):
        common_nodes = np.intersect1d(self.nodes, factor.nodes)
        nr_common_nodes = common_nodes.size
        nr_common_nodes_values_combinations = np.prod([len(self.network.values[common_node]) for common_node in common_nodes])

        self.sort(common_nodes)
        factor.sort(common_nodes)

        nodes = np.append(self.nodes, factor.nodes[nr_common_nodes:])

        products1 = self.products_with_equal_common_nodes(nr_common_nodes_values_combinations)
        products2 = factor.products_with_equal_common_nodes(nr_common_nodes_values_combinations)
        products = [[products1[i, e] * products2[i] for e in xrange(len(products1[0]))] for i in xrange(len(products1))]
        products = np.vstack([product.reshape((-1, 1)) for value_products in products for product in value_products])

        probs = Factor.get_probs(self, factor, products, nr_common_nodes, nodes)

        return Factor(nodes, probs, self.network)

    def marginalize(self, node):
        self.sort([node])
        nr_values = len(self.network.values[node])
        nr_equal_values_in_column = len(self.probs) / nr_values

        indices_list = [[i * nr_equal_values_in_column + e for i in xrange(nr_values)] for e in xrange(nr_equal_values_in_column)]
        sums = np.array([self.probs[indices, -1] for indices in indices_list]).astype(np.float)
        sums = np.sum(sums, axis = 1).reshape((-1, 1))

        values = self.probs[:nr_equal_values_in_column, 1:-1]

        probs = np.hstack((values, sums))

        return Factor(self.nodes[1:], probs, self.network)

    ## TODO
    def reduce(self, observed, nodes_in_observed):
        self.sort(nodes_in_observed)

        nodes_in_observed = [(node, observed[node]) for node in nodes_in_observed]
        nodes_in_observed_values_data = [(self.network.values[node].index(value), len(self.network.values[node]))
                                            for node, value in nodes_in_observed]

        nr_rows = len(self.probs)
        bounds = [0, nr_rows]
        for index, nr_values in nodes_in_observed_values_data:
            nr_equal_values_in_column = nr_rows / nr_values
            bounds[0] += index * nr_equal_values_in_column
            bounds[1] -= (nr_values - index - 1) * nr_equal_values_in_column
            nr_rows /= nr_values

        probs = self.probs[bounds[0] : bounds[1], len(nodes_in_observed):]

        nodes = self.nodes[len(nodes_in_observed):]

        return Factor(nodes, probs, self.network)

    def sort(self, nodes):
        nodes_columns_indices = np.flatnonzero(np.in1d(self.nodes, nodes))

        for i, index in enumerate(nodes_columns_indices):
            if i != index:
                self.nodes[[i, index]] = self.nodes[[index, i]]
                self.probs[:, [i, index]] = self.probs[:, [index, i]]

        dtype = [(str(i), 'S10') for i in xrange(self.nr_nodes + 1)]
        self.probs = np.array(map(tuple, self.probs), dtype = dtype)

        self.probs = np.sort(self.probs, order = map(str, xrange(self.nr_nodes)))

        self.probs = np.array(map(list, self.probs))

    def products_with_equal_common_nodes(self, nr_common_nodes_values_combinations):
        nr_equal_common_nodes_values_in_columns = len(self.probs) / nr_common_nodes_values_combinations
        bounds_list = [[i * nr_equal_common_nodes_values_in_columns, (i + 1) * nr_equal_common_nodes_values_in_columns] for i in
                       xrange(nr_common_nodes_values_combinations)]
        return np.array([self.probs[bounds[0] : bounds[1], -1] for bounds in bounds_list]).astype(np.float)

    def normalize(self):
        prob_column = self.probs[:,-1].astype(np.float).reshape((-1, 1))
        prob_column = prob_column / np.sum(prob_column)

        probs = np.hstack((self.probs[:,:-1].reshape((-1, self.nr_nodes)), prob_column))

        return Factor(self.nodes, probs, self.network)
