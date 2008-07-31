# Puts together the ONN and the GA to train the network.

module GAONN

  require "gsl"

  require File.expand_path(File.dirname(__FILE__)) + "/onn"
  include OscillatorNeuralNetwork

  class GAONN

    # GA parameter default values
    DEFAULT_POPULATION_SIZE = 100
    DEFAULT_NUM_GENS        = 100
    DEFAULT_MUTATION_RATE   = 0.4

    :attr_accessor init_node_data

    # Creates a new GAONN instance
    #   node_data:     a matrix containing the hidden and output layer node data
    #   connections:   a connections matrix for entire network
    #   inputs:        a list of matrices of input state vectors, one matrix per input node
    #   seed:          a PRNG seed governing all calls to rand for this simulation
    #   mutation_rate: the rate at which mutations occur in the GA
    def initialize(node_data,connections,inputs,seed,mutation_rate)
      @init_node_data = node_data.clone
      @initial_conns = connections.clone
      @input_list = inputs.clone
      @num_inputs = inputs.size2
      @num_outputs = outputs.size2
      @mutation_rate = mutation_rate
      srand(seed)
    end

    # Returns best node_data and its fitness
    def train
      ga = GeneticSearch.new(self, DEFAULT_POPULATION_SIZE, DEFAULT_NUM_GENS, DEFAULT_MUTATION_RATE)
      return ga.run
    end

    # Error/fitness function (for now based on "orthogonal amplitude" idea).
    #   chromosome: a chromosome from the GA (for now, a node_data matrix)
    def fitness(chromosome)

      @outputs = GSL::Matrix.alloc(@input_list.size,@num_outputs)
      @net = ONN.new(@input_list,chromosome,@conns,@num_outputs,@num_inputs)

      @net.eval_over_time
      amps, freqs = @net.fourier_analyze
      @outputs.set_row(0,amps.to_gv)

      1..@input_list.size do |index|
        @net.set_input(index)
        @net.eval_over_time
        amps, freqs = @net.fourier_analyze
        @outputs.set_row(index,amps.to_gv)
      end

      norm_error = eval_error
      fitness = 1 - norm_error
      return fitness
    end

    # Evaluates the error of the @outputs currently stored
    def eval_error
      # TODO implement
      return rand
    end

  end

  class GeneticSearch

    attr_reader :population_size
    attr_reader :max_generation
    attr_reader :curr_generation
    attr_reader :population
    attr_reader :mutation_rate

    # Creates new GeneticSearch instance 
    #   gaonn:           the gaonn 
    #   population_size: the size of the population of potential solutions 
    #   generations:     the number of generations to run the GA 
    #   mutation_rate:   the mutation rate
    def initialize(gaonn, population_size, generations, mutation_rate)
      @population_size = population_size 
      @max_generation = generations
      @curr_generation = 0
      @population = []     
      @gaonn = gaonn
      @mutation_rate = mutation_rate
    end
    
    # Runs a genetic algorithm, returning the best chromosome (node list) on completion
    #  Algorithm summary:
    #    1. Choose initial population (randomly or otherwise distributed)
    #    2. Evaluate the fitness of each individual in the population
    #    3. Repeat until @max_generation:
    #      1. Select best-ranking individuals to reproduce 
    #      2. Breed new generation through mutation and give birth to offspring 
    #      3. Evaluate the individual fitnesses of the offspring
    #      4. Replace worst ranked part of population with offspring
    #    4. Return the fittest member of the population and its fitness
    def run
      generate_initial_population 
      @max_generation.times do
        selected_to_breed = selection                # evaluate current population 
        offsprings = reproduction(selected_to_breed) # generate the population for this new generation
        replace_worst_ranked(offsprings)
      end
      @population.sort! { |a, b| @gaonn.fitness(b) <=> @gaonn.fitness(a)}
      return @population[0], @ga_onn.fitness(@population[0])
    end
  
    # Generates population by adding random uniform noise to the given node_data.
    #   variation_radius: describes the range of noise that will be added
    def generate_initial_population(variation_radius)
      orig_data = @gaonn.init_node_data
      @population_size.times do
        @population << perturb_matrix(orig_data,variation_radius)
      end
    end

    # Adds uniform noise to a matrix
    def perturb_matrix(mat, radius)
      mat2 = mat.clone
      mat2.collect! { |entry| entry + (rand(2*radius)-radius) }
      return mat2
    end

    def mutate(mat, radius)
      mat.collect! { |entry|
        if rand < @mutation_rate
          entry + (rand(2*radius)-radius) 
        end
      }
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
      @population.sort! { |a, b| @gaonn.fitness(b) <=> @gaonn.fitness(a)}
      best_fitness = @gaonn.fitness(@population[0])
      worst_fitness = @gaonn.fitness(@population.last)
      acum_fitness = 0
        if best_fitness - worst_fitness > 0
          @population.each do |chromosome| 
          acum_fitness += @gaonn.fitness(chromosome)
        end
      end

      selected_to_breed = []
      ((2*@population_size)/3).times do 
        selected_to_breed << select_random_individual(acum_fitness)
      end
      selected_to_breed
    end
    
    # we combine each pair of selected chromosome using the method 
    # chromosome.reproduce
    #
    # the reproduction will also call the chromosome.mutate method with 
    # each member of the population. you should implement chromosome.mutate
    # to only change (mutate) randomly. e.g. you could effectivly change the
    # chromosome only if 
    #     rand < ((1 - chromosome.fitness) * 0.4)
    def reproduction(selected_to_breed)
      selected_to_breed.each do |individual|
        mutate(individual,mutation_radius)
      end
      return selected_to_breed
    end
    
    # replace worst ranked part of population with offspring
    def replace_worst_ranked(offsprings)
      size = offsprings.length
      @population = @population[0..((-1*size)-1)] + offsprings
    end
    
    def select_random_individual(acum_fitness)
      select_random_target = acum_fitness * rand
      local_acum = 0
      @population.each do |chromosome|
        local_acum += @gaonn.fitness(chromosome)
        return chromosome if local_acum >= select_random_target
      end
    end
  end

end
