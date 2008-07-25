# Testing file which tests a single network with different inputs and plots
# the various amplitudes and frequencies of the outputs over time.
require File.expand_path(File.dirname(__FILE__)) + "/../lib/onn"
include OscillatorNeuralNetwork
require 'test/unit'
require 'gsl'

class TestMultipleInputs < Test::Unit::TestCase

  def setup
    @node_data = GSL::Matrix[[0.7,0,1,0,0,0],[0.3,0.1,0,0,0,1],[0.8,0.1,0,0,0,1],[0.2,0.1,0,0,0,2],[0.7,0.1,0,0,0,2]]
    @conns = GSL::Matrix[[0,0.4,0.4,0,0],[0,0,0.2,0.4,-0.6],[0,0.2,0,-0.6,0.4],[0,0,0,0,0],[0,0,0,0,0]]
    @inputs = GSL::Vector.linspace(0.1,3.0,30)
    @num_outputs = 2
  end

  def test_inputs
    node3amps = []
    node4amps = []
    node3freqs = []
    node4freqs = []
    input_index = 1

    @inputs.each do |input|
      @node_data[0][0] = input
      @net = GAONN.new(@node_data,@conns,@num_outputs,2000)
      @net.eval_over_time
      ret_vals3 = @net.fourier_analyze(3)
      assert_equal(ret_vals3.size,2)
      amp3 = ret_vals3[0]
      freq3 = ret_vals3[1]
      assert_kind_of(Float,amp3)
      assert_kind_of(Float,freq3)
      node3amps << amp3
      node3freqs << freq3
      assert_equal(input_index,node3amps.size)
      assert_equal(input_index,node3freqs.size)
      ret_vals4 = @net.fourier_analyze(4)
      assert_equal(2,ret_vals4.size)
      amp4 = ret_vals4[0]
      freq4 = ret_vals4[1]
      assert_kind_of(Float,amp4)
      assert_kind_of(Float,freq4)
      node4amps << amp4 
      node4freqs << freq4 
      assert_equal(node4amps.size,input_index)
      assert_equal(node4freqs.size,input_index)
      input_index += 1
    end

    GSL::graph(@inputs,node3amps.to_gv,node4amps.to_gv,"-S 16 -m -2 -T png -C -X 'input a' -Y 'output amplitudes' -L 'Output Amplitudes vs. Input a' > output_amps.png")
    GSL::graph(@inputs,node3freqs.to_gv,node4freqs.to_gv,"-S 16 -m -2 -T png -C -X 'input a' -Y 'output frequencies' -L 'Output Frequencies vs. Input a' > output_freqs.png")
  end


end
