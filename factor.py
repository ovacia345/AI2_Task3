import numpy as np

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
    def get_probs(factor1, factor2, prob_column, nr_common_nodes):
        values1 = factor1.probs[:, :-1]  # Includes common_nodes columns
        values2 = factor2.probs[:, nr_common_nodes:-1]  # Excludes common_nodes columns

        if values1[:, nr_common_nodes:].size > 0 and values2.size > 0:
            nr_rows = prob_column.size

            probs = [np.repeat(values1, nr_rows / values1.shape[0], axis=0)]
            probs += [np.tile(values2.T, nr_rows / values2.shape[0]).T]
        elif values2.size > 0:
            probs = [np.repeat(values1, values2.shape[0] / values1.shape[0], axis=0)]
            probs += [values2]
        else:
            probs = [values1]
        probs += [prob_column]

        return np.hstack(probs)


    def reduce(self, nodes_in_observed, observed):
        self.sort(nodes_in_observed)

        nodes_in_observed = [(node, observed[node]) for node in nodes_in_observed]
        nodes_in_observed_values_data = [(self.network.values[node].index(value), len(self.network.values[node]))
                                         for node, value in nodes_in_observed]

        nr_rows = self.probs.shape[0]
        nr_rows_list = [nr_rows]*len(nodes_in_observed)
        for i, (_, nr_values) in enumerate(nodes_in_observed_values_data):
            nr_rows_list[i:] = [nr_rows_list[i] / nr_values]*len(nr_rows_list[i:])

        bounds = [np.sum([index * nr_rows_list[i] for i, (index, _) in enumerate(nodes_in_observed_values_data)])]
        bounds += [nr_rows - np.sum([(nr_values - (index + 1)) * nr_rows_list[i] for i, (index, nr_values) in enumerate(nodes_in_observed_values_data)])]

        probs = self.probs[bounds[0]:bounds[1], len(nodes_in_observed):]

        nodes = self.nodes[len(nodes_in_observed):]

        return Factor(nodes, probs, self.network)

    def times(self, factor):
        common_nodes = np.intersect1d(self.nodes, factor.nodes)
        nr_common_nodes = common_nodes.size
        nr_common_nodes_values_combinations = np.prod([len(self.network.values[common_node]) for common_node in common_nodes])

        self.sort(common_nodes)
        factor.sort(common_nodes)

        nodes = np.append(self.nodes, factor.nodes[nr_common_nodes:])

        products1 = self.products_with_equal_common_nodes(nr_common_nodes_values_combinations)
        products2 = factor.products_with_equal_common_nodes(nr_common_nodes_values_combinations)
        products_list = [[products1[i, e] * products2[i] for e in xrange(products1.shape[1])] for i in xrange(products1.shape[0])]
        products_columns = [product.reshape((-1, 1)) for products in products_list for product in products]
        products = np.vstack(products_columns)

        probs = Factor.get_probs(self, factor, products, nr_common_nodes)

        return Factor(nodes, probs, self.network)

    def marginalize(self, node):
        self.sort([node])
        nr_values = len(self.network.values[node])
        nr_equal_values_in_column = self.probs.shape[0] / nr_values

        indices_list = [[i * nr_equal_values_in_column + e for i in xrange(nr_values)] for e in xrange(nr_equal_values_in_column)]
        sums_list = [self.probs[indices, -1] for indices in indices_list]
        sums = np.array(sums_list).astype(np.float)
        sums = np.sum(sums, axis = 1).reshape((-1, 1))

        values = self.probs[:nr_equal_values_in_column, 1:-1]

        probs = np.hstack((values, sums))

        return Factor(self.nodes[1:], probs, self.network)

    def normalize(self, query):
        self.sort(query)

        prob_column = self.probs[:,-1].astype(np.float).reshape((-1, 1))
        prob_column = prob_column / np.sum(prob_column)

        probs = np.hstack((self.probs[:,:-1].reshape((-1, self.nr_nodes)), prob_column))

        return Factor(self.nodes, probs, self.network)


    def sort(self, nodes):
        for i, node in enumerate(nodes):
            index = np.flatnonzero(self.nodes == node)[0]
            if i != index:
                self.nodes[[i, index]] = self.nodes[[index, i]]
                self.probs[:, [i, index]] = self.probs[:, [index, i]]

        dtype = [(str(i), 'S10') for i in xrange(self.nr_nodes)] + [('prob', np.float)]
        self.probs = np.array(map(tuple, self.probs), dtype = dtype)

        self.probs = np.sort(self.probs, order = map(str, xrange(self.nr_nodes)))

        self.probs = np.array(map(list, self.probs))

    def products_with_equal_common_nodes(self, nr_common_nodes_values_combinations):
        nr_equal_common_nodes_values_in_columns = self.probs.shape[0] / nr_common_nodes_values_combinations
        bounds_list = [[i * nr_equal_common_nodes_values_in_columns, (i + 1) * nr_equal_common_nodes_values_in_columns] for i in
                       xrange(nr_common_nodes_values_combinations)]
        products = [self.probs[bounds[0]:bounds[1], -1] for bounds in bounds_list]
        return np.array(products).astype(np.float)
