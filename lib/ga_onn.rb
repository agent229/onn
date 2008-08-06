# Puts together the ONN and the GA to train the network.

module GAONN

  require "gsl"

  require File.expand_path(File.dirname(__FILE__)) + "/onn"
  include OscillatorNeuralNetwork

  class GA

    # GA parameter default values
    DEFAULT_POPULATION_SIZE   = 2
    DEFAULT_NUM_GENS          = 2
    DEFAULT_MUTATION_RATE     = 0.5
    DEFAULT_MUTATION_RADIUS   = 1.0 
    DEFAULT_SEED              = 0

    attr_accessor :node_data
    attr_accessor :mutation_rate
    attr_accessor :mutation_radius
    attr_accessor :conns
    attr_accessor :input_list
    attr_accessor :num_inputs
    attr_accessor :num_outputs
    attr_accessor :rng

    # Creates a new GAONN
    #   node_data:       a matrix containing the hidden and output layer node data
    #   connections:     a connections matrix for entire network
    #   inputs:          a list of matrices of input state vectors, one matrix per input node
    #   num_outputs:     number of outputs
    #   seed:            a PRNG seed governing all calls to rand for this simulation
    #   population_size: the size of the population of potential solutions 
    #   generations:     the number of generations to run the GA 
    #   mutation_rate:   the rate at which mutations occur in the GA
    #   mutation_radius: a description of how much to mutate things
    def initialize(node_data,connections,inputs,num_outputs,seed=DEFAULT_SEED,population_size=DEFAULT_POPULATION_SIZE,generations=DEFAULT_NUM_GENS,mutation_rate=DEFAULT_MUTATION_RATE,mutation_radius=DEFAULT_MUTATION_RADIUS)
      @population_size = population_size 
      raise "0/negative population size" if @population_size <= 0
      @max_generation = generations
      raise "0/negative num gens" if @max_generation <= 0
      @population = []     
      @mutation_rate = mutation_rate
      @mutation_radius = mutation_radius
      @node_data = node_data.clone
      @conns = connections.clone
      raise "Bad input nodedata/conns length mismatch" if @node_data.size1 != @conns.size1
      @input_list = inputs
      @num_inputs = inputs[0].size1
      @num_outputs = num_outputs
      @rng = GSL::Rng.alloc(GSL::Rng::MT19937,seed)
    end
    
    # Runs a genetic algorithm, returning the best chromosome (node list) on completion
    #  Algorithm summary:
    #    1. Choose initial population (randomly or otherwise distributed)
    #    2. Evaluate the fitness of each individual in the population
    #    3. Repeat until @max_generation:
    #      1. Select best-ranking individuals to reproduce (copy)
    #      2. Replace worst ranked part of population with offspring
    #      3. Mutate current population
    #    4. Return the fittest member of the population and its fitness
    def run
      generate_initial_population 
      @max_generation.times do |generation|
        offsprings = selection 
        raise "wrong output from selection" if offsprings.class != Array
        offsprings.each do |child|
          raise "wrong output from selection" if child.class != AmpChromosome
        end
        replace_worst_ranked(offsprings)
        reproduction
        puts "finished gen # " + generation.to_s
      end
      return best_chromosome, best_chromosome.normalized_fitness
    end
  
    # Generates population by adding random uniform noise to the given node_data.
    def generate_initial_population
      @population_size.times do
        @population << AmpChromosome.new(self)
      end
      raise "bad pop generation" if @population.size != @population_size
      @population.each do |chrom|
        raise "bad pop generation" if chrom.class != AmpChromosome
      end
    end

    # Select the best chromosome in the population
    def best_chromosome
      the_best = @population[0]
      @population.each do |chromosome|
        the_best = chromosome if chromosome.fitness > the_best.fitness
      end
      return the_best
    end
    
    # Selection is the stage of a genetic algorithm in which individual 
    # genomes are chosen from a population for later breeding. 
    # There are several generic selection algorithms, such as 
    # tournament selection and roulette wheel selection. We implemented the
    # latest.
    # 
    # Steps:
    # 1. the fitness function is evaluated for each individual, providing fitness values
    # 2. the population is sorted by descending fitness values.
    # 3. a random number r is chosen. r is between 0 and the accumulated normalized value 
    #    (all the normalized fitness values added togheter).
    # 4. the selected individual is the first one whose accumulated normalized value 
    #    (its is normalized value plus the normalized values of the chromosomes prior it) greater than r.
    # 5. we repeat steps 4 and 5, 2/3 times the population size.    
    def selection
      @population.sort! { |a, b| b.fitness <=> a.fitness }
      best_fitness = @population[0].fitness
      worst_fitness = @population.last.fitness
      acum_fitness = 0
      if best_fitness - worst_fitness > 0
        @population.each do |chromosome| 
          chromosome.normalized_fitness = (chromosome.fitness-worst_fitness)/(best_fitness-worst_fitness)
          acum_fitness += chromosome.normalized_fitness
        end
      else
        @population.each { |chromosome| chromosome.normalized_fitness = 1 }
      end
      selected_to_breed = []
      ((2*@population_size)/3).round.times do 
        selected_to_breed << select_random_individual(acum_fitness)
      end
      return selected_to_breed
    end
   
    # Mutates all with same chance, reproduces (by copying) those that are good.
    def reproduction
      @population.each do |chromosome|
        AmpChromosome.mutate(chromosome,@mutation_radius,@mutation_rate,@rng)
      end
    end
    
    # replace worst ranked part of population with offspring
    def replace_worst_ranked(offsprings)
      size = offsprings.length
      @population = @population[0..((-1*size)-1)] + offsprings
      raise "wrong number replaced" if @population.size != @population_size
    end
    
    def select_random_individual(acum_fitness)
      select_random_target = acum_fitness * @rng.uniform
      local_acum = 0
      @population.each do |chromosome|
        local_acum += chromosome.normalized_fitness
        return chromosome if local_acum >= select_random_target
      end
    end
  end

  # Describies a chromosome of the GA, which is currently node_data only and judges its
  # fitness based on the output amplitudes and orthoganality
  class AmpChromosome

    attr_accessor :node_data
    attr_accessor :normalized_fitness

    # Initialize a chromosome
    def initialize(ga)
      @ga = ga
      @node_data = perturb_matrix(ga.node_data) 
      raise "bad perturbation" if @node_data.size1 != ga.node_data.size1
      raise "bad perturbation" if @node_data.size2 != ga.node_data.size2
    end

    # Adds uniform noise to a matrix in a given radius
    def perturb_matrix(mat)
      mat2 = mat.clone
      mat2.collect! { |entry| entry + (@ga.rng.uniform*2*@ga.mutation_radius-@ga.mutation_radius) }
      row_index = 0
      mat2.col(0).each do |a_val|
        if a_val < 0
          mat2[row_index,0] = a_val.abs
        end
        row_index += 1
      end
      return mat2
    end

    def self.mutate(chrom,mutation_radius,mutation_rate,rng)
      mutation_radius = mutation_radius * (1-chrom.normalized_fitness)
      mat = chrom.node_data
      changed_flag = false
      mat.collect! { |entry|
        if chrom.normalized_fitness && rng.uniform < ((1-chrom.normalized_fitness) * mutation_rate)
          changed_flag = true
          entry + (rng.uniform(2*mutation_radius)-mutation_radius) 
        else entry
        end
      }
      @fitness = nil if changed_flag==true
    end

    # Error/fitness function (for now based on "orthogonal amplitude" idea).
    #   chromosome: a chromosome from the GA (for now, a node_data matrix)
    def fitness
   
      return @fitness if @fitness

      outputs = GSL::Matrix.alloc(@ga.num_inputs,@ga.num_outputs)
      @net = ONN.new(@ga.input_list,@node_data,@ga.conns,@ga.num_outputs,@ga.num_inputs)
      beg_ind = @net.nodes.size-@ga.num_outputs
      end_ind = @net.nodes.size

      @net.eval_over_time
      amps = []
      freqs = []
      for index in beg_ind...end_ind 
        amps_i, freqs_i = @net.fourier_analyze(index)
        amps << amps_i
      end
      outputs.set_row(0,amps.to_gv)

      0..@ga.input_list.size do |index|
        @net.set_input(index)
        @net.eval_over_time
        amps = []
        freqs = []
        for index in beg_ind...end_ind 
          amps_i, freqs_i = @net.fourier_analyze(index)
          amps << amps_i
        end
        outputs.set_row(index,amps.to_gv)
      end

      error = eval_fitness(outputs)
      @fitness = 1 - 1/error
      throw "fitness out of range!" if @fitness < 0 
      throw "fitness out of range!" if @fitness > 1
      return @fitness
    end

    # Evaluates the error of the @outputs currently stored
    def eval_fitness(outputs_mat)
      outputs = outputs_mat.clone
      det = Math::sqrt(GSL::Linalg::LU.det(outputs.transpose*outputs))
      return det.abs
    end

  end

end
