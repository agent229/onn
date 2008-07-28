# "Driver" file which puts together the ONN and the GA to train the network.

require "gsl"

require File.expand_path(File.dirname(__FILE__)) + "/onn"
include OscillatorNeuralNetwork

require File.expand_path(File.dirname(__FILE__)) + "/ga"
include GeneticSearch

class GAONN

  def initialize(node_data,connections,inputs,outputs,seed,mutation_rate)
    @node_data = node_data.clone
    @conns = connections.clone
    @inputs = inputs.clone
    @outputs = outputs.clone
    @num_inputs = inputs.size2
    @num_outputs = outputs.size2
    @seed = seed
    @mutation_rate = mutation_rate
    srand(seed)
  end

  def train
    @inputs.each_row do |input_vals|
    input_index = 0
      input_vals.each do |val|
        @node_data[input_index][0] = val
        @net = ONN.new(@node_data,@conns,@num_outputs)
        @net.eval_over_time
        input_index += 1
      end
    end
  end

  # TODO fix up
  # Error weighting function. Calculates a Euclidean-style weighted error measure of the output.
  #   expected: the expected/desired result (GSL::Matrix of data)
  def fitness
    w_err = 0.0
    result_amps = []
    result_freqs = []
    @nodes.size-@num_outputs..@nodes.size do |index|
      result_amps_i, result_freqs_i = fourier_analyze(index)
      result_amps << result_amps_i
      result_freqs << result_freqs_i
    end
    result_amps.each_index do |node_index|
      amp_term = result_amps[node_index] - amp_from_vec(expected[node_index])
      freq_term = result_freqs[node_index] - freq_from_vec(expected[node_index])
      w_err += GSL::hypot(amp_term, freq_term)
    end
    w_err /= result_amps.length
    return w_err
  end
end

