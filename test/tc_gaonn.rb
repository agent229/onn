## GAONN testing file

require File.expand_path(File.dirname(__FILE__)) + "/../lib/ga_onn"
include GAONN
require 'test/unit'

class TestGAONN < Test::Unit::TestCase

  def setup
    @node_data = GSL::Matrix[[0.7,0,1,0,0,0],[0.3,0.1,0,0,0,1],[0.8,0.1,0,0,0,1],[0.2,0.1,0,0,0,2],[0.7,0.1,0,0,0,2]]
    @conns = GSL::Matrix[[0,0.4,0.4,0,0],[0,0,0.2,0.4,-0.6],[0,0.2,0,-0.6,0.4],[0,0,0,0,0],[0,0,0,0,0]]
    @inputs = []
    10.times do |index|
      input_set = GSL::Matrix.calloc(1,5)
      input_set[0,0] = rand*2
      input_set[0,2] = 1
      @inputs << input_set
    end
    @ga = GA.new(@node_data,@conns,@inputs,2)
  end

  def test_run
    best_chrom, fitness = @ga.run
    puts "best_chrom"
    puts best_chrom
    puts "fitness" + fitness.to_s
  end

end
