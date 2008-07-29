# Testing file for the genetic algorithm OscillatorNetwork
require File.expand_path(File.dirname(__FILE__)) + "/../lib/onn" 
include OscillatorNetwork
require 'test/unit'
require 'gsl'

class TestNetwork < Test::Unit::TestCase

  def setup
#     node_data = GSL::Matrix[[4*GSL::M_PI*GSL::M_PI,0,1.0,0,0,0]]
#     conns = GSL::Matrix[[0]]
#     @net= Network.new(node_data,conns,1)


#    node_data = GSL::Matrix[[4*GSL::M_PI*GSL::M_PI,0,1.0,0.2,0,0],[20.0,0,1.0,0.2,0,0],[40.0,0.05,0,0,0,1],[30.0,0.05,0,0,0,1]]
#    conns = GSL::Matrix[[0,0,0.1,0.1],[0,0,0.1,0.1],[0,0,0,0],[0,0,0,0]]
#    @net = Network.new(node_data,conns,2)

  #  node_data = GSL::Matrix[[0.9,0,1.0,0.2,0,0],[0.3,0.05,0,0,0,1],[0.7,0.05,0,0,0,1]]
  #  conns = GSL::Matrix[[0,-0.1,-0.1],[0,0,0],[0,0,0]]
  #  @net = Network.new(node_data,conns,2)

  # node_data = GSL::Matrix[[0.4,0,1,0.1,0,0],[0.2,0.2,0,0,0,1],[0.3,0.2,0,0,0,1],[0.6,0.1,0,0,0,2],[1.2,0.1,0,0,0,2]]
  # conns = GSL::Matrix[[0,0.8,0.2,0,0],[0,0,0,0.6,-0.4],[0,0,0,-0.4,0.6],[0,0,0,0,0],[0,0,0,0,0]]
  # @net = Network.new(node_data,conns,2) 

    node_data = GSL::Matrix[[0.70000001,0,1.0000000000001,0,0,0],[0.30000001,0.10000000001,0,0,0,1],[0.80000001,0.10000000001,0,0,0,1],[0.20000001,0.100000001,0,0,0,2],[0.70000001,0.10000000001,0,0,0,2]]
    conns = GSL::Matrix[[0,0.4000000001,0.4000000001,0,0],[0,0,0.2000000001,0.400000001,-0.600000001],[0,0.200000001,0,-0.60000001,0.40000001],[0,0,0,0,0],[0,0,0,0,0]]
    @net = Network.new(node_data,conns,2) 
    
  #  node_data = GSL::Matrix[[0.7,0,1,0.1,0,0],[0.7,0.1,0,0,0,1],[0.7,0.2,0,0,0,1],[0.7,0.1,0,0,0,2]]
  #  conns = GSL::Matrix[[0,0.4,0.3,0],[0,0,0,0.4],[0,0,0,0.4],[0,0,0,0]]
  #  @net = Network.new(node_data,conns,1)
  end

  def test_init
    assert_kind_of(Array,@net.nodes)

    @net.nodes.each do |node|
      assert_kind_of(OscillatorNode,node)
    end

    assert_kind_of(GSL::Matrix,@net.connections)
    assert_equal(@net.curr_time,0.0)
  end

  def test_init_neuron
    assert_kind_of(Hash,@net.nodes[0].out_conns)
    assert_kind_of(Array,@net.nodes[0].input_sum_terms)
  end

  def test_eval
    @net.eval_over_time
    amps = freqs = []
    @net.nodes.each_index do |index|
      @net.plot_x_over_time(index)
      return_vals = @net.fourier_analyze(index)
      puts "node #{index} amp: " + return_vals[0].to_s
      puts "node #{index} freq: " + return_vals[1].to_s
    end
#    expected = GSL::Matrix[[0,0,0,0,0,0],[0,0,0,0,0,0]] # TODO put in expected results
#    err = @net.weighted_error(expected)
  end

end
