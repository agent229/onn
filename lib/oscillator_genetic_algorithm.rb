# The OscillatorGeneticAlgorithm module implements the GeneticSearch, ParamChromosome, 
# and WeightChromosome classes. The GeneticSearch is a generic class, and can be used to solved 
# any kind of problems. The GeneticSearch class performs a stochastic search 
# of the solution of a given problem.
# 
# The chromosome is "problem specific". For the oscillator neural network, there are
# two types of chromosomes: WeightChromosome keeps track of connection weights and 
# modifies them to train the network, and ParamChromosome keeps track of one of the
# parameters of the wave equation (amplitude, phase, or frequency) and adjusts these
# to train the network.
# 
# Based upon the genetic algorithm in the ai4r library.

module OscillatorGeneticAlgorithm
  
  #   This class is used to automatically:
  #   
  #     1. Choose initial population
  #     2. Evaluate the fitness of each individual in the population
  #     3. Repeat
  #           1. Select best-ranking individuals to reproduce (depending on chromosome type)
  #           2. Breed new generation through crossover and mutation (genetic operations) 
  #              and give birth to offspring (depending on chromosome type)
  #           3. Evaluate the individual fitnesses of the offspring
  #           4. Replace worst ranked part of population with offspring
  #     4. Until termination
  class GeneticSearch
   
    attr_accessor :population

    # Creates new GeneticSearch
    #   initial_population_size: the size of the population to evolve
    #   generations: the number of generations to evolve
    #   seed: a random number seed for this entire run, for repeatability reasons
    #   chrom_size: describes the size of any given chromosome
    def initialize(initial_population_size, generations, chrom_size, seed)
      # Seed random number gen, initialize fields
      srand(seed)                       
      @population_size = initial_population_size
      @max_generation = generations
      @generation = 0
      @chrom_size = chrom_size
    end
    
    # Runs the genetic algorithm as described above, returning the best chromosome on completion
    def run
      generate_initial_population                     # generate initial population 
      @max_generation.times do
        selected_to_breed = selection                 # evaluate current population 
        offsprings = reproduction(selected_to_breed)  # generate the population for this new generation
        replace_worst_ranked(offsprings)              # replace the worst members
      end
      return best_chromosome
    end
   
    # Generates the initial population of chromosomes according to that chromosome's seed method
    def generate_initial_population
     @population = []
     @population_size.times do
       chromosome = WeightChromosome.new(@chrom_size) # For now, just a WeightChromosome
       population << chromosome.seed                  # Uniformly distributed values
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
    # 3. the fitness values ar then normalized. (highest fitness gets 1, lowest fitness gets 0). 
    #    the normalized value is stored in the "normalized_fitness" attribute of the chromosomes.
    # 4. a random number r is chosen. r is between 0 and the accumulated normalized value 
    #    (all the normalized fitness values added togheter).
    # 5. the selected individual is the first one whose accumulated normalized value 
    #    (its is normalized value plus the normalized values of the chromosomes prior it) greater than r.
    # 6. we repeat steps 4 and 5, 2/3 times the population size.    
    def selection
      @population.sort! { |a, b| b.fitness <=> a.fitness}
      best_fitness = @population[0].fitness
      worst_fitness = @population.last.fitness
      acum_fitness = 0
      if best_fitness-worst_fitness > 0
      @population.each do |chromosome| 
        chromosome.normalized_fitness = (chromosome.fitness - worst_fitness)/(best_fitness-worst_fitness)
        acum_fitness += chromosome.normalized_fitness
      end
      else
        @population.each { |chromosome| chromosome.normalized_fitness = 1}  
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
    #     rand < ((1 - chromosome.normalized_fitness) * 0.4)
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
        local_acum += chromosome.normalized_fitness
        return chromosome if local_acum >= select_random_target
      end
    end
  end

  # A WeightChromosome relates to the weights between the nodes in the network. This can be
  # best expressed as a matrix of connections, where 0 represents no connection and anything
  # else represents the weight, where the ijth element designates a connection between node i and j.
  # There should be no crossover here as it doesn't make sense for this type of net; mutation only.
  # In this case, all connections are free game for mutation.... should probably be changed eventually
  class WeightChromosome
    
    attr_accessor :data
    attr_accessor :normalized_fitness
    attr_accessor :size
   
    # Data will be a size x size  matrix, where size is the number of nodes
    def initialize(size)
      @size = size 
      @data = GSL::Matrix.alloc(@size) # Allocate an empty matrix for this set of weights
    end
    
    def fitness
      return @fitness if @fitness
      #TODO write propagation so this works... fitness weights.... etc
      return @fitness
    end

    # Random noise added to weights
    #   chromosome: the chromosome to mutate
    #   mutation_rate: a tunable parameter describing chance of mutation
    #   mutation_size: a tunable parameter describing the maximum size of the mutation
    #   pct_weights_mutated: a tunable parameter describing the percentage of the weights that gets mutated if
    #                        the chromosome is selected for mutation
    def self.mutate(chromosome, mutation_rate, mutation_size, pct_weights_mutated)
      if chromosome.normalized_fitness && rand < ((1 - chromosome.normalized_fitness) * mutation_rate)
        1.upto(pct_weights_mutated*chromosome.size*chromosome.size) do
          data[rand(chromosome.size),rand(chromosome.size)] += (rand(2*mutation_size) - mutation_size) 
        end
      @fitness = nil
      end
    end
    
    # Reproduction here involves no crossover, simply duplication. 
    def self.reproduce(a, b)
      offspring = WeightChromosome.new(a.size)
      0.upto(a.size) do |row|
        0.upto(a.size) do |col|
          offspring.data[row,col]=a.data[row,col]
        end
      end
      return offspring
    end
    
    # Initializes an individual solution for the initial population.
    # Right now, simply fills the data matrix with evenly distributed values on [0,1)
    def self.seed
      0.upto(@size-1) do |row|
        0.upto(@size-1) do |col|
          @data[row,col] = rand
        end
      end
    end

    # TODO want to use this? how would i set cost matrix?
    def self.set_cost_matrix(costs)
      @@costs = costs
    end
  end
end

