# GA testing file
require '~/Documents/SFI/NN/onn/lib/onn_genetic_algorithm'
include ONNGeneticAlgorithm
require '~/Documents/SFI/NN/onn/lib/onn'
include OscillatorNeuralNetwork
require 'test/unit'

class TestGA < Test::Unit::TestCase

  def setup
    @conns = [[0, 1], [0, 0]]
    @net = GAOscillatorNeuralNetwork.new(1.0, 2, 45)
    @ga = GeneticSearch.new(@net, 10, 10, 0.4)
  end

  def test_init
    assert_equal(@ga.population_size, 10)
    assert_equal(@ga.max_generation, 10)
    assert_equal(@ga.curr_generation, 0)
    assert_equal(@ga.network,@net)
    assert_equal(@ga.population,[])
    assert_equal(@ga.mutation_rate,0.4)
  end

  def test_generate_initial_population
    @ga.generate_initial_population
    assert_equal(@ga.population.length,@ga.population_size)
    @ga.population.each do |nodelist|
      assert_kind_of(Array,nodelist)
      nodelist.each do |node|
        assert_kind_of(OscillatorNeuron,node)
      end
    end
  end

  def test_run
    best = @ga.run
    assert_kind_of(Array,best)
    best.each do |node|
      assert_kind_of(OscillatorNeuron,node)
    end
  end

end
