# A very simple sample simulation using the onn and oscillator_genetic_algorithm

require("rbgsl") # For matrices
require "onn"
include OscillatorNeuralNetwork 

# create data -- three oscillators

connections = GSL::Matrix[[0, 0.5, 0], [0, 0, 0.5], [0, 0, 0]]
states = GSL::Matrix[[0, 0, 0], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]

# instantiate network
net = GeneticAlgorithmONN.new(connections, states) 

# train
