"""
@Author: Joris van Vugt, Moira Berens

Implementation of the variable elimination algorithm for AISPAML assignment 3

"""
import numpy as np

class VariableElimination():

    def __init__(self, network):
        self.network = network
        self.addition_steps =  0
        self.multiplication_steps = 0

        self.factors = np.array([])

    def run(self, query, observed, elim_order):
        netProbs = self.network.probabilities
        netValues = self.network.values
        self.network.values = {node: sorted(netValues[node]) for node in netValues}

        for variable in self.network.nodes:
            factor = self.makeFactor(variable, observed)
            if factor.nrNodes > 0:
                self.factors = np.append(self.factors, factor)

        for variable in elim_order:
            idx = np.array([i for i, factor in enumerate(self.factors) if variable in factor.nodes])
            if variable != elim_order[0]:
                idx = np.unique(np.append(idx, np.array([len(self.factors) - 1])))
            variableFactors = self.factors[idx]
            for i in xrange(len(variableFactors) - 1):
                product = variableFactors[i].times(variableFactors[i + 1])
                variableFactors[i + 1] = product

            marginalization = variableFactors[-1].marginalize(variable)

            self.factors = np.append(self.factors, 1)
            self.factors[-1] = marginalization
            self.factors = np.delete(self.factors, idx)
        print
        print "Result:"
        print self.factors[0].nodes
        print self.factors[0].probs
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

    def makeFactor(self, variable, observed):
        nodes = np.array([variable] + self.network.parents[variable])
        probs = self.network.probabilities[variable].values
        factor = Factor(nodes, probs, self.network)
        if np.any(np.in1d(nodes, observed.keys())):
            if len(nodes) > np.sum(np.in1d(nodes, observed.keys())):
                factor.reduce(observed)
                return factor
            else:
                return Factor([], [], None)
        else:
            return factor



class Factor():

    def __init__(self, nodes, probs, network):
        self.nodes = nodes
        self.nrNodes = len(self.nodes)

        self.probs = probs

        self.network = network

    def times(self, factor):
        common_variables = np.intersect1d(self.nodes, factor.nodes)
        nr_common_variables = len(common_variables)
        nr_common_variables_values_combinations = np.prod([len(self.network.values[common_variable]) for common_variable in common_variables])


        self.sort(common_variables)
        factor.sort(common_variables)


        nodes = np.append(self.nodes, factor.nodes[nr_common_variables:])


        products1 = self.probs_with_equal_common_variables(nr_common_variables_values_combinations)
        products2 = factor.probs_with_equal_common_variables(nr_common_variables_values_combinations)

        products = np.hstack([[products1[i, e] * products2[i] for i in xrange(len(products1))]
                              for e in xrange(len(products1[0]))]).reshape((-1, 1))

        probs = get_probs(self, factor, products, nr_common_variables, nodes)


        return Factor(nodes, probs, self.network)

    def marginalize(self, variable):
        self.sort([variable])
        nr_values = len(self.network.values[variable])
        nr_times_values_in_column = len(self.probs) / nr_values

        indices_list = [[i * nr_times_values_in_column + e for i in xrange(nr_values)] for e in xrange(nr_times_values_in_column)]
        sums = np.array([self.probs[indices, -1] for indices in indices_list]).astype(np.float)
        sums = np.sum(sums, axis = 1).reshape((-1, 1))

        values = self.probs[:len(self.probs) / nr_values, 1:-1]

        probs = np.hstack((values, sums))

        return Factor(self.nodes[1:], probs, self.network)

    def reduce(self, observed_variables):
        keys = observed_variables.keys()

        self.sort(keys)

        for key in keys:
            values = np.array(self.network.values[key])
            value_index = np.flatnonzero(values == observed_variables[key])[0]
            nr_equal_values_in_column = len(self.probs) / len(values)
            bounds = [value_index * nr_equal_values_in_column, (value_index + 1) * nr_equal_values_in_column]

            self.probs = self.probs[bounds[0] : bounds[1], 1:]

        self.nodes = self.nodes[len(keys):]
        self.nrNodes -= len(keys)

    def sort(self, variables):
        variable_columns_indices = np.flatnonzero(np.in1d(self.nodes, variables))

        for i, index in enumerate(variable_columns_indices):
            if i != index:
                self.nodes[[i, index]] = self.nodes[[index, i]]
                self.probs[:, [i, index]] = self.probs[:, [index, i]]

        dtype = [(str(i), 'S10') for i in xrange(self.nrNodes)] + [('value', np.float)]
        self.probs = np.array(map(tuple, self.probs), dtype = dtype)

        self.probs = np.sort(self.probs, order = map(str, xrange(self.nrNodes)))

        self.probs = np.array(map(list, self.probs))

    def probs_with_equal_common_variables(self, nr_common_variables_values_combinations):
        nr_rows_in_set_rows = len(self.probs) / nr_common_variables_values_combinations
        bounds_list = [[i * nr_rows_in_set_rows, (i + 1) * nr_rows_in_set_rows] for i in
                       xrange(nr_common_variables_values_combinations)]
        return np.array([self.probs[bounds[0] : bounds[1], -1] for bounds in bounds_list]).astype(np.float)


def get_probs(factor1, factor2, products, nr_common_variables, nodes):
    values1 = factor1.probs[:, :-1]
    values2 = factor2.probs[:, nr_common_variables:-1]

    if values1[:, nr_common_variables:].size != 0 and values2.size != 0:
        nr_rows = 1
        for node in nodes:
            nr_rows *= len(factor1.network.values[node])

        probs = [np.repeat(values1, nr_rows / len(values1), axis=0)]
        probs += [np.tile(values2.T, nr_rows / len(values2)).T]
    elif values1[:, nr_common_variables:].size != 0:
        probs = [values1]
    elif values2.size != 0:
        probs = [np.repeat(values1, len(values2) / len(values1), axis=0)]
        probs += [values2]
    probs += [products]

    return np.hstack(probs)
