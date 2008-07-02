# Testing file for the genetic algorithm ONN
require '~/Documents/SFI/NN/onn/lib/onn'
include OscillatorNeuralNetwork
require 'test/unit'

class TestONN < Test::Unit::TestCase

  def setup
    @net1 = GAOscillatorNeuralNetwork.new(1.0, 2, 45)
    @net2 = GAOscillatorNeuralNetwork.new(0.7, 3, 420)
  end

  def test_init
    assert_equal(@net1.t_step,1.0)
    assert_equal(@net1.num_nodes,2)
    assert_equal(@net1.seed,45)

    assert_equal(@net2.t_step,0.7)
    assert_equal(@net2.num_nodes,3)
    assert_equal(@net2.seed,420)
  end

  def test_nodes
    @net1.set_nodes([[0.75, 0.56],[0.45, 0.34]]) 
    assert_equal(@net1.nodes.length,2)

    @net1.nodes.each do |node|
      assert_kind_of(OscillatorNeuron,node)
    end

    assert_equal(@net1.nodes[0].natural_state[:freq],0.75) 
    assert_equal(@net1.nodes[0].natural_state[:phase],0.56) 
    assert_equal(@net1.nodes[1].natural_state[:freq],0.45) 
    assert_equal(@net1.nodes[1].natural_state[:phase],0.34) 

    @net2.set_nodes([[0.12, 0.78], [0.34, 0.89], [0.56, 0.90]]) 
    assert_equal(@net2.nodes.length,3)

    @net2.nodes.each do |node|
      assert_kind_of(OscillatorNeuron,node)
    end

    assert_equal(@net2.nodes[0].natural_state[:freq],0.12) 
    assert_equal(@net2.nodes[0].natural_state[:phase],0.78) 
    assert_equal(@net2.nodes[1].natural_state[:freq],0.34) 
    assert_equal(@net2.nodes[1].natural_state[:phase],0.89) 
    assert_equal(@net2.nodes[2].natural_state[:freq],0.56) 
    assert_equal(@net2.nodes[2].natural_state[:phase],0.90)

    # test set_connections
    @net1.set_connections([[0, 1],[1, 0]])
    assert_equal(@net1.nodes[0].in_conns.length,1)
    assert_equal(@net1.nodes[0].in_conns[@net1.nodes[1]],1)
    assert_equal(@net1.nodes[0].out_conns.length,1)
    assert_equal(@net1.nodes[0].out_conns[@net1.nodes[1]],1)
    assert_equal(@net1.nodes[1].in_conns.length,1)
    assert_equal(@net1.nodes[1].in_conns[@net1.nodes[0]],1)
    assert_equal(@net1.nodes[1].out_conns.length,1)
    assert_equal(@net1.nodes[1].out_conns[@net1.nodes[0]],1)

    @net2.set_connections([[0, 1, 0],[1, 0, 1],[0, 0, 0]])
    assert_equal(@net2.nodes[0].in_conns.length,1)
    assert_equal(@net2.nodes[0].in_conns[@net2.nodes[1]],1)
    assert_equal(@net2.nodes[0].out_conns.length,1)
    assert_equal(@net2.nodes[0].out_conns[@net2.nodes[1]],1)
    assert_equal(@net2.nodes[1].out_conns.length,2)
    assert_equal(@net2.nodes[1].in_conns.length,1)
    assert_equal(@net2.nodes[1].in_conns[@net2.nodes[0]],1)
    assert_equal(@net2.nodes[1].out_conns[@net2.nodes[0]],1)
    assert_equal(@net2.nodes[1].out_conns[@net2.nodes[2]],1)
    assert_equal(@net2.nodes[2].in_conns.length,1)
    assert_equal(@net2.nodes[2].in_conns[@net2.nodes[1]],1)
    assert_equal(@net2.nodes[2].out_conns.length,0)
  end

  def test_generate_random_node_data
    data1 = @net1.generate_random_node_data
    assert_equal(data1.length,2)

    data1.each do |node|
      node.each do |data|
        assert_operator(0,:<=,data)
        assert_operator(1,:>=,data)
      end
    end

    data2 = @net2.generate_random_node_data
    assert_equal(data2.length,3)

    data2.each do |node|
      node.each do |data|
        assert_operator(0,:<=,data)
        assert_operator(1,:>=,data)
      end
    end
  end

  def test_eval
    # TODO write this
  end
end

class TestOscillatorNeuron < Test::Unit::TestCase
 
  def setup
    @hash1 = {:phase => 0.45, :freq => 0.78, :fake_var => 0.9}
    @node1 = OscillatorNeuron.new(@hash1)
  end

  def test_init_neuron
    assert_equal(@hash1,@node1.curr_state)
    assert_equal(@hash1,@node1.natural_state)
    assert_equal(nil,@node1.next_state)
    assert_kind_of(Hash,@node1.in_conns)
    assert_kind_of(Hash,@node1.out_conns)
    assert_kind_of(Array,@node1.input_sum_terms)
  end

  def test_update_and_reset_state
    @node1.input_sum_terms = [1,1]
    @node1.update_state
    assert_equal(-0.77,@node1.curr_state[:phase])
    assert_equal(nil,@node1.next_state)

    @node1.reset_state
    assert_equal(@node1.natural_state,@node1.curr_state)
  end

  def test_propagate
    # TODO write this.... or maybe put in the entire network tests
  end

end
