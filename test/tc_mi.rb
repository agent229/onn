# Testing file which tests a single network with different inputs and plots
# the various amplitudes and frequencies of the outputs over time.
require File.expand_path(File.dirname(__FILE__)) + "/../lib/onn"
include OscillatorNeuralNetwork
require 'test/unit'
require 'gsl'

class TestMultipleInputs < Test::Unit::TestCase

  def setup
    @node_data = GSL::Matrix[[0.3,0.1,0,0,0,1],[0.8,0.1,0,0,0,1],[0.2,0.1,0,0,0,2],[0.7,0.1,0,0,0,2]]
    @conns = GSL::Matrix[[0,0.4,0.4,0,0],[0,0,0.2,0.4,-0.6],[0,0.2,0,-0.6,0.4],[0,0,0,0,0],[0,0,0,0,0]]
    @inputs = [] 
    50.times do |index|
      input_set = GSL::Matrix.calloc(1,5)
      input_set[0,0] = rand*4
      input_set[0,2] = rand*2
      @inputs << input_set
    end

    @num_outputs = 2
    @num_inputs = 1
  end

  def test_inputs
    node3amps = []
    node4amps = []
    node3freqs = []
    node4freqs = []
    input_index = 1

    @net = ONN.new(@inputs,@node_data,@conns,@num_outputs,@num_inputs)

    @net.eval_over_time

    ret_vals3 = @net.fourier_analyze(3)
    amp3 = ret_vals3[0]
    node3amps << amp3
    ret_vals4 = @net.fourier_analyze(4)
    amp4 = ret_vals4[0]
    node4amps << amp4 

    for index in 1...@inputs.size 
      @net.set_input(index)
      @net.eval_over_time
      ret_vals3 = @net.fourier_analyze(3)
      amp3 = ret_vals3[0]
      node3amps << amp3
      ret_vals4 = @net.fourier_analyze(4)
      amp4 = ret_vals4[0]
      node4amps << amp4 
    end

    puts node3amps
    puts node4amps

    input_a_vals = []
    @inputs.each do |input|
      input_a_vals << input[0,0]
    end

    GSL::graph(input_a_vals.to_gv.sqrt,node3amps.to_gv,node4amps.to_gv,"-S 16 -m -2 -T png -C -X 'input frequency' -Y 'output amplitudes' -L 'Output Amplitudes vs. Input Frequency' > output_amps.png")
  end

end
