"""
@Author: Joris van Vugt, Moira Berens

Implementation of the variable elimination algorithm for AISPAML assignment 3

"""
import numpy as np

class VariableElimination():

    def __init__(self, network):
        self.network = network
        self.network.values = {node: sorted(self.network.values[node]) for node in self.network.values}

        self.addition_steps =  0
        self.multiplication_steps = 0

    ## Finished
    def run(self, query, observed, elim_order):
        factors = np.array([])
        for node in self.network.nodes:
            factor = self.makeFactor(node, observed)
            if factor.nr_nodes > 0:
                factors = np.append(factors, factor)

        for node in elim_order:
            node_factors_indices = [i for i, factor in enumerate(factors) if node in factor.nodes]

            node_factors = factors[node_factors_indices]
            for i in xrange(node_factors.size - 1):
                product = node_factors[i].times(node_factors[i + 1])
                node_factors[i + 1] = product

            marginalization = node_factors[-1].marginalize(node)

            factors = np.append(factors, marginalization)
            factors = np.delete(factors, node_factors_indices)

        print
        print "Result:"
        print factors[0].nodes
        print factors[0].probs

        """
        Use the variable elimination algorithm to find out the probability
        distribution of the query variable given the observed variables

        Input:
            query:      The query variable
            observed:   A dictionary of the observed variables {variable: value}
            elim_order: Either a list specifying the elimination ordering
                        or a function that will determine an elimination ordering
                        given the network during the run

        Output: A variable holding the probability distribution
                for the query variable

        """

    ## Finished
    def makeFactor(self, node, observed):
        nodes = np.array([node] + self.network.parents[node])
        probs = self.network.probabilities[node].values

        nodes_in_observed = nodes[np.flatnonzero(np.in1d(nodes, observed.keys()))]
        if nodes_in_observed.size > 0:
            if nodes.size > nodes_in_observed.size:
                factor = Factor(nodes, probs, self.network)
                return factor.reduce(nodes_in_observed)
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

    ## Finished
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

        probs = get_probs(self, factor, products, nr_common_nodes, nodes)

        return Factor(nodes, probs, self.network)

    ## Finished
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

    def reduce(self, observed_nodes):
        keys = observed_nodes.keys()

        self.sort(keys)

        for key in keys:
            values = np.array(self.network.values[key])
            value_index = np.flatnonzero(values == observed_nodes[key])[0]
            nr_equal_values_in_column = len(self.probs) / len(values)
            bounds = [value_index * nr_equal_values_in_column, (value_index + 1) * nr_equal_values_in_column]

            self.probs = self.probs[bounds[0] : bounds[1], 1:]

        self.nodes = self.nodes[len(keys):]
        self.nr_nodes -= len(keys)

    ## Finished
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

    ## Finished
    def products_with_equal_common_nodes(self, nr_common_nodes_values_combinations):
        nr_equal_common_nodes_values_in_columns = len(self.probs) / nr_common_nodes_values_combinations
        bounds_list = [[i * nr_equal_common_nodes_values_in_columns, (i + 1) * nr_equal_common_nodes_values_in_columns] for i in
                       xrange(nr_common_nodes_values_combinations)]
        return np.array([self.probs[bounds[0] : bounds[1], -1] for bounds in bounds_list]).astype(np.float)

    ## Finished
def get_probs(factor1, factor2, products, nr_common_nodes, nodes):
    values1 = factor1.probs[:, :-1] # Includes common_nodes columns
    values2 = factor2.probs[:, nr_common_nodes:-1] # Excludes common_nodes columns

    if values1[:, nr_common_nodes:].size != 0 and values2.size != 0:
        nr_rows = np.prod([len(factor1.network.values[node] for node in nodes)])

        probs = [np.repeat(values1, nr_rows / len(values1), axis=0)]
        probs += [np.tile(values2.T, nr_rows / len(values2)).T]
    elif values1[:, nr_common_nodes:].size != 0:
        probs = [values1]
    elif values2.size != 0:
        probs = [np.repeat(values1, len(values2) / len(values1), axis=0)]
        probs += [values2]
    else:
        probs = [values1]
    probs += [products]

    return np.hstack(probs)
