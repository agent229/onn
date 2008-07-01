# Testing file for the genetic algorithm ONN
require '~/Documents/SFI/NN/onn/lib/onn'
include OscillatorNeuralNetwork
require 'test/unit'

class TestONN < Test::Unit::TestCase

  def setup
    @net = GAOscillatorNeuralNetwork.new(1.0, 2, 45, [[0, 1],[1, 0]])
  end

  def test_init
    assert_equal(@net.t_step,1.0)
    assert_equal(@net.num_nodes,2)
    assert_equal(@net.connections,[[0, 1],[1, 0]])
  end

  def test_set_nodes
    @net.set_nodes([[0.75, 0.56],[0.45, 0.34]]) 
    assert_equal(@net.nodes.length,2)
    assert_kind_of(OscillatorNeuron,@net.nodes[1])
    assert_kind_of(OscillatorNeuron,@net.nodes[0])
    assert_equal(@net.nodes[0].natural_state[:natural_freq],0.75) 
    assert_equal(@net.nodes[0].natural_state[:natural_phase],0.56) 
    assert_equal(@net.nodes[1].natural_state[:natural_freq],0.45) 
    assert_equal(@net.nodes[1].natural_state[:natural_phase],0.34) 
    # test_update_connections
    @net.update_connections(@net.connections)
    assert_equal(@net.nodes[0].in_conns[@net.nodes[1]],1)
    assert_equal(@net.nodes[0].in_conns[@net.nodes[1]],1)
    assert_equal(@net.nodes[0].out_conns[@net.nodes[1]],1)
    assert_equal(@net.nodes[0].out_conns[@net.nodes[1]],1)
  end

end
