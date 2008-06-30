# GA testing file
require '~/Documents/SFI/NN/onn/lib/onn_genetic_algorithm'
include ONNGeneticAlgorithm
require '~/Documents/SFI/NN/onn/lib/onn'
include OscillatorNeuralNetwork
require 'test/unit'

class TestGA < Test::Unit::TestCase

  def setup
    @net = GAOscillatorNeuralNetwork.new(1.0, 2, 45, [[0, 1],[1, 0]])
    @ga = GeneticSearch.new(@net, 10, 10)
  end

  def test_init
    assert_equal(@ga.population_size, 10)
    assert_equal(@ga.max_generation, 10)
    assert_equal(@ga.curr_generation, 0)
    assert_equal(@ga.network,@net)
    assert_equal(@ga.population,[])
  end

  def test_generate_initial_population
    @ga.generate_initial_population
    assert_equal(@ga.population.length,@ga.population_size)
    assert_kind_of(Array,@ga.population[0])
    assert_kind_of(OscillatorNeuron,@ga.population[0][0])
  end

  def test_run

  end

end
