# The OscillatorGeneticAlgorithm module implements the GeneticSearch class. 
# The GeneticSearch class performs a stochastic search of the solution of a given problem.
#
# In this case, the "chromosome" is a list of OscillatorNeuron objects, storing their states.
# 
# Based upon the genetic algorithm in the ai4r library.

module ONNGeneticAlgorithm

  # This class is used to run a genetic algorithm search of solutions for an ONN training
  class GeneticSearch

    attr_reader :population_size
    attr_reader :max_generation
    attr_reader :curr_generation
    attr_reader :population
    attr_reader :network

    # Creates new GeneticSearch instance 
    #   network: an initialized GAOscillatorNeuralNetwork
    #   population_size: the size of the population of potential solutions 
    #   generations:     the number of generations to run the GA 
    def initialize(network, population_size, generations)
      # Initialize fields
      @population_size = population_size 
      @max_generation = generations
      @curr_generation = 0
      @population = [] # Will store population_size lists of OscillatorNeuron objects
      @network = network
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
    #    4. Return the fittest member of the population
    def run
      generate_initial_population                     # generate initial population 
      @max_generation.times do
        selected_to_breed = selection                 # evaluate current population 
        offsprings = reproduction(selected_to_breed)  # generate the population for this new generation
        replace_worst_ranked(offsprings)              # replace the worst members
      end
      return best_chromosome
    end
   
    # Generates the initial population randomly (uniformly distributed properites)
    # with a given connections matrix
    def generate_initial_population
      @population_size.times do
        data_arr = @network.generate_random_node_data
        @network.set_nodes(data_arr)
        @network.update_connections(@network.connections)
        population << @network.nodes 
      end
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
      @population.sort! { |a, b| b.fitness <=> a.fitness}
      best_fitness = @population[0].fitness
      worst_fitness = @population.last.fitness
      acum_fitness = 0
      if best_fitness-worst_fitness > 0
        @population.each do |chromosome| 
          acum_fitness += chromosome.fitness
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
      offsprings = []
      0.upto(selected_to_breed.length/2-1) do |i|
        offsprings << chromosome.reproduce(selected_to_breed[2*i], selected_to_breed[2*i+1])
      end
      @population.each do |individual|
        chromosome.mutate(individual)
      end
      return offsprings
    end
    
    # replace worst ranked part of population with offspring
    def replace_worst_ranked(offsprings)
      size = offsprings.length
      @population = @population [0..((-1*size)-1)] + offsprings
    end
    
    # select the best chromosome in the population
    def best_chromosome
      the_best = @population[0]
      @population.each do |chromosome|
        the_best = chromosome if chromosome.fitness > the_best.fitness
      end
      return the_best
    end
    
  private 
    def select_random_individual(acum_fitness)
      select_random_target = acum_fitness * rand
      local_acum = 0
      @population.each do |chromosome|
        local_acum += chromosome.fitness
        return chromosome if local_acum >= select_random_target
      end
    end
  end

end

