import numpy as np

# values: the values of the nodes, e.g. 'no' or 'yes'.
# prob_column / products / sums: the column of probabilities.
# probs: the array where values and prob_column are combined.
class Factor():

    def __init__(self, nodes = np.array([]), probs = np.array([]), network = None):
        self.nodes = nodes
        self.nr_nodes = self.nodes.size

        self.probs = probs
        self.nr_rows = self.probs.shape[0]

        self.network = network

    def __str__(self):
        string = "Nodes: " + str(self.nodes) + "\n"
        string += "Probabilities:\n" + str(self.probs) + "\n"
        return string

    # This function computes the product of all the factors in the factors parameter.
    @staticmethod
    def product(factors):
        nr_multiplication_steps = factors.size - 1
        for i in xrange(nr_multiplication_steps):
            factors[1] = factors[0].times(factors[1])
            factors = np.delete(factors, 0)
        return factors[0], nr_multiplication_steps

    # When doing factor multiplication, the resulting factor might have more nodes.
    # This function returns the probs of such a bigger factor.
    @staticmethod
    def get_probs(factor1, factor2, prob_column, nr_common_nodes):
        values1 = factor1.probs[:, :-1]  # Includes common nodes columns
        values2 = factor2.probs[:, nr_common_nodes:-1]  # Excludes common nodes columns

        if factor1.nr_nodes > nr_common_nodes and factor2.nr_nodes > nr_common_nodes:
            nr_rows = prob_column.size

            probs = [np.repeat(values1, nr_rows / factor1.nr_rows, axis=0)]
            probs += [np.tile(values2.T, nr_rows / factor2.nr_rows).T]
        elif factor2.nr_nodes > nr_common_nodes:
            probs = [np.repeat(values1, factor2.nr_rows / factor1.nr_rows, axis=0)]
            probs += [values2]
        else:
            probs = [values1]
        probs += [prob_column]

        return np.hstack(probs)


    # This function reduces a factor given the nodes that are in the factor that are observed
    def reduce(self, nodes_in_observed, observed):
        self.sort(nodes_in_observed)

        nr_nodes_in_observed = nodes_in_observed.size
        nodes = self.nodes[nr_nodes_in_observed:]

        nodes_in_observed = [(node, observed[node]) for node in nodes_in_observed]
        nodes_in_observed_values_data = [(self.network.values[node].index(value), len(self.network.values[node]))
                                         for node, value in nodes_in_observed]
                                         # List of tuples (index of value in values, number of values)

        nr_rows = self.nr_rows
        bounds = [0, nr_rows]
        for index, nr_values in nodes_in_observed_values_data:
            nr_rows /= nr_values
            bounds[0] += index * nr_rows
            bounds[1] -= (nr_values - index - 1) * nr_rows

        probs = self.probs[bounds[0]:bounds[1], nr_nodes_in_observed:]

        return Factor(nodes, probs, self.network)

    # This function computes the multiplication of this factor with another factor
    def times(self, factor):
        common_nodes = np.intersect1d(self.nodes, factor.nodes)
        nr_common_nodes = common_nodes.size
        nr_common_nodes_values_combinations = np.prod([len(self.network.values[common_node])
                                                       for common_node in common_nodes])

        self.sort(common_nodes)
        factor.sort(common_nodes)

        nodes = np.append(self.nodes, factor.nodes[nr_common_nodes:])

        products1 = self.probabilities_with_equal_common_nodes(nr_common_nodes_values_combinations)
        products2 = factor.probabilities_with_equal_common_nodes(nr_common_nodes_values_combinations)
        products_list = [[products1[i, e] * products2[i] for e in xrange(products1.shape[1])]
                         for i in xrange(nr_common_nodes_values_combinations)]
        products_columns = [product.reshape((-1, 1)) for products in products_list for product in products]
        products = np.vstack(products_columns)

        probs = Factor.get_probs(self, factor, products, nr_common_nodes)

        return Factor(nodes, probs, self.network)

    def marginalize(self, node):
        self.sort([node])
        nr_values = len(self.network.values[node])
        nr_equal_values_in_column = self.nr_rows / nr_values

        values = self.probs[:nr_equal_values_in_column, 1:-1]

        indices_list = [[i * nr_equal_values_in_column + e for i in xrange(nr_values)]
                        for e in xrange(nr_equal_values_in_column)]
        sums_list = [self.probs[indices, -1] for indices in indices_list]
        sums = np.array(sums_list).astype(np.float)
        sums = np.sum(sums, axis = 1).reshape((-1, 1))

        probs = np.hstack((values, sums))

        return Factor(self.nodes[1:], probs, self.network)

    def normalize(self, query):
        self.sort([query])

        values = self.probs[:, :-1].reshape((-1, self.nr_nodes))

        prob_column = self.probs[:,-1].astype(np.float).reshape((-1, 1))
        prob_column = prob_column / np.sum(prob_column)

        probs = np.hstack((values, prob_column))

        return Factor(self.nodes, probs, self.network)


    # All functions that are used in variable_elim.py do not change the factor.
    # This function does change the factor, because it is only used in factor.py.
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

    # This function returns the parts that will be multiplied in the times function.
    def probabilities_with_equal_common_nodes(self, nr_common_nodes_values_combinations):
        nr_equal_common_nodes_values = self.nr_rows / nr_common_nodes_values_combinations
        bounds_list = [[i * nr_equal_common_nodes_values, (i + 1) * nr_equal_common_nodes_values]
                       for i in xrange(nr_common_nodes_values_combinations)]
        products = [self.probs[bounds[0]:bounds[1], -1] for bounds in bounds_list]
        return np.array(products).astype(np.float)
