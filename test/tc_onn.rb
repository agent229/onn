# Testing file for the genetic algorithm ONN
require File.expand_path(File.dirname(__FILE__)) + "/../lib/onn" 
include OscillatorNeuralNetwork
require 'test/unit'
require 'gsl'

class TestONN < Test::Unit::TestCase

  def setup
#     node_data = GSL::Matrix[[4*GSL::M_PI*GSL::M_PI,0,1.0,0,0,0]]
#     conns = GSL::Matrix[[0]]
#     @net= GAONN.new(node_data,conns,1)


#    node_data = GSL::Matrix[[4*GSL::M_PI*GSL::M_PI,0,1.0,0.2,0,0],[20.0,0,1.0,0.2,0,0],[40.0,0.05,0,0,0,1],[60.0,0.05,0,0,0,1]]
#    conns = GSL::Matrix[[0,0,0.1,0.1],[0,0,0.1,0.1],[0,0,0,0],[0,0,0,0]]
#    @net = GAONN.new(node_data,conns,2,0,0.01,15000)

  #  node_data = GSL::Matrix[[0.9,0,1.0,0.2,0,0],[0.3,0.05,0,0,0,1],[0.7,0.05,0,0,0,1]]
  #  conns = GSL::Matrix[[0,-0.1,-0.1],[0,0,0],[0,0,0]]
  #  @net = GAONN.new(node_data,conns,2)

  # node_data = GSL::Matrix[[0.4,0,1,0.1,0,0],[0.2,0.2,0,0,0,1],[0.3,0.2,0,0,0,1],[0.6,0.1,0,0,0,2],[1.2,0.1,0,0,0,2]]
  # conns = GSL::Matrix[[0,0.8,0.2,0,0],[0,0,0,0.6,-0.4],[0,0,0,-0.4,0.6],[0,0,0,0,0],[0,0,0,0,0]]
  # @net = GAONN.new(node_data,conns,2) 

    node_data = GSL::Matrix[[0.7,0,1,0,0,0],[0.3,0.1,0,0,0,1],[0.8,0.1,0,0,0,1],[0.2,0.1,0,0,0,2],[0.7,0.1,0,0,0,2]]
    conns = GSL::Matrix[[0,0.4,0.4,0,0],[0,0,0.2,0.4,-0.6],[0,0.2,0,-0.6,0.4],[0,0,0,0,0],[0,0,0,0,0]]
    @net = GAONN.new(node_data,conns,2) 
    
  #  node_data = GSL::Matrix[[0.7,0,1,0.1,0,0],[0.7,0.1,0,0,0,1],[0.7,0.2,0,0,0,1],[0.7,0.1,0,0,0,2]]
  #  conns = GSL::Matrix[[0,0.4,0.3,0],[0,0,0,0.4],[0,0,0,0.4],[0,0,0,0]]
  #  @net = GAONN.new(node_data,conns,1)
  end

  def test_init
    assert_equal(@net.seed,0)
    assert_kind_of(Array,@net.nodes)

    @net.nodes.each do |node|
      assert_kind_of(OscillatorNeuron,node)
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
    @net.nodes.each_index do |index|
      @net.plot_x_over_time(index)
      @net.fourier_analyze(index)
    end
#    expected = GSL::Matrix[[0,0,0,0,0,0],[0,0,0,0,0,0]] # TODO put in expected results
#    err = @net.weighted_error(expected)
  end

end
