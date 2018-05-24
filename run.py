"""
@Author: Joris van Vugt, Moira Berens

Entry point for testing the variable elimination algorithm

"""
from read_bayesnet import BayesNet
from variable_elim import *
from heuristics import *
import datetime

if __name__ == '__main__':
    # the class BayesNet represents a Bayesian network from a .bif file
    # in several variables
    net = BayesNet('child.bif')

    # these are the variables that should be used for variable elimination
    print 'values', net.values
    print 'probabilities', net.probabilities
    print 'parents', net.parents
    print 'nodes', net.nodes


    # Make your variable elimination code in a seperate file: 'variable_elim'.
    # you can call this file as follows:
    ve = VariableElimination(net)

    # If variables are known beforehand, you can represent them in the following way:
    evidence = {'Sick': 'no', 'CO2Report': '<7.5', 'Grunting': 'yes'}

    # determine you heuristics before you call the run function. This can be done in this file or in a seperate file
    # The heuristics either specifying the elimination ordering (list) or it is a function that determines the elimination ordering
    # given the network. An simple example is:
    query = 'Age'
    elim_order = min_weight(net, evidence, query)

    #call the elimination ordering function for example as follows:
    result = ve.run(query, evidence, elim_order)

    print "\nResult:"
    print("Query: " + query)
    print("Evidence: " + str(evidence))
    print("Probabilities: \n" + str(result.probs))
    print("\nWall clock time: " + str(ve.wall_clock_time))
    print("Number of multiplication steps: " + str(ve.nr_multiplication_steps))


 
