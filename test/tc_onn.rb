# Testing file for the genetic algorithm ONN
require File.expand_path(File.dirname(__FILE__)) + "/../lib/onn" 
include OscillatorNeuralNetwork
require 'test/unit'
require 'gsl'

class TestONN < Test::Unit::TestCase

  def setup
    node_data = GSL::Matrix[[0.5,0.2,1,0,0,0],[0.7,0.3,0,0,0,1]]
    conns = GSL::Matrix[[0,0.75],[0,0]]
    @net = GAONN.new(node_data,conns) 
  end

  def test_init
    assert_equal(@net.seed,0)
    assert_kind_of(Array,@net.nodes)

    @net.nodes.each do |node|
      assert_kind_of(OscillatorNeuron,node)
    end

    assert_kind_of(GSL::Matrix,@net.connections)
    assert_equal(@net.nodes[0].out_conns.length,1)
    assert_equal(@net.nodes[1].out_conns.length,0)

    assert_equal(@net.curr_time,0.0)
  end

  def test_init_neuron
    assert_kind_of(GSL::Vector,@net.nodes[0].state_vector)
    assert_kind_of(Hash,@net.nodes[0].out_conns)
    assert_kind_of(Array,@net.nodes[0].input_sum_terms)
  end


end
