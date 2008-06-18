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
   
    srand(1) # For repeatability, can change/record seed
    attr_accessor :population

    # TODO see where chromosome initialization comes from, how to set it to weight/param type...  
    def initialize(initial_population_size, generations)
      @population_size = initial_population_size
      @max_generation = generations
      @generation = 0
    end
    
    # Runs the genetic algorithm as described above, then returns the best chromosome
    def run
      generate_initial_population                    #generate initial population 
      @max_generation.times do
        selected_to_breed = selection                #evaluates current population 
        offsprings = reproduction selected_to_breed  #generate the population for this new generation
        replace_worst_ranked offsprings
      end
      return best_chromosome
    end
    
    def generate_initial_population
     @population = []
     @population_size.times do
       population << chromosome.seed # TODO which chromosome is it using? how do i set it?
     end
    end
    
    # selection is the stage of a genetic algorithm in which individual 
    # genomes are chosen from a population for later breeding. 
    # there are several generic selection algorithms, such as 
    # tournament selection and roulette wheel selection. we implemented the
    # latest.
    # 
    # steps:
    # 
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

  # A ParamChromosome describes one of the parameters of an oscillator (with one entry
  # for each oscillator in the network). This could be natural phase, amplitude, or frequency.
  # Thus the number of data in the chromosome should be the number of oscillators in the network.
  # The chromosome data will be normalized on [0,1) so that it may be used for different purposes;
  # the user can simply multiply by the appropriate scale once they recieve the data.
  class ParamChromosome
    
    srand(1) # For repeatability, can change/record seed
    attr_accessor :data
    attr_accessor :normalized_fitness
    
    def initialize(data)
      @data = data
    end
    
    # The fitness function quantifies the optimality of a solution 
    # (that is, a chromosome) in a genetic algorithm so that that particular 
    # chromosome may be ranked against all the other chromosomes. 
    # In this case, the fitness function is essentially feeding the parameters into
    # the network simulation and measuring the error compared to the desired output. 
    def fitness
      return @fitness if @fitness
      #TODO
      return @fitness
    end

    # For ParamChromosomes, mutation means random, small noise added with some chance to
    # any given parameter in the chromosome to maintain genetic diversity. Because the
    # parameters are normalized in the GA, the noise added definitely needs to be in (-1,1)
    # but should be quite small. This will perturb the system to avoid minima. 
    def self.mutate(chromosome)
      data = chromosome.data
      0.upto(data.length-1)
        data = chromosome.data
        if chromosome.normalized_fitness && rand < ((1 - chromosome.normalized_fitness) * 0.3)
          data[index] += rand(0.25)-0.125
          chromosome.data = data
          @fitness = nil
        end
      end
    end
    
    # Reproduction is used to vary the programming of a chromosome or 
    # chromosomes from one generation to the next. There are several ways to
    # combine two chromosomes: One-point crossover, Two-point crossover,
    # "Cut and splice", edge recombination, and more. 
    # 
    # The method is usually dependant of the problem domain.
    # In this case, we have implemented edge recombination, wich is the 
    # most used reproduction algorithm for the Travelling salesman problem.
    # TODO see if this is suitable for this purpose...
    def self.reproduce(a, b)
      data_size = @@costs[0].length
      available = []
      0.upto(data_size-1) { |n| available << n }
      token = a.data[0]
      spawn = [token]
      available.delete(token)
      while available.length > 0 do 
        #Select next
        if token != b.data.last && available.include?(b.data[b.data.index(token)+1])
          next_token = b.data[b.data.index(token)+1]
        elsif token != a.data.last && available.include?(a.data[a.data.index(token)+1])
          next_token = a.data[a.data.index(token)+1] 
        else
          next_token = available[rand(available.length)]
        end
        #Add to spawn
        token = next_token
        available.delete(token)
        spawn << next_token
        a, b = b, a if rand < 0.4
      end
      return ParamChromosome.new(spawn)
    end
    
    # Initializes an individual solution (chromosome) for the initial 
    # population. Generates random numbers in [0,1) for each parameter. 
    def self.seed
      data_size = @@costs[0].length # need other way to get data size...
      0.upto(data_size-1) 
        seed << rand(1)
      end 
      return ParamChromosome.new(seed)
    end

    def self.set_cost_matrix(costs)
      @@costs = costs
    end
  end

  # A WeightChromosome relates to the weights between the nodes in the network. This can be
  # best expressed as a matrix of connections, where 0 represents no connection and anything
  # else represents the weight, where the ijth element designates a connection between node i and j.
  # There should be no crossover here as it doesn't make sense for this type of net; mutation only.
  class WeightChromosome
    
    srand(1) # For repeatability, can change/record seed
    attr_accessor :data
    attr_accessor :normalized_fitness
    
    def initialize(data)
      @data = data
    end
    
    # Should be essentialy same as fitness function for ParamChromosome
    def fitness
      return @fitness if @fitness
      #TODO
      return @fitness
    end

    # Random noise added to weights, but this time only to nonzero weights. Zero weights represent
    # lack of any connection and should be left alone completely. We are only evolving the non-
    # zero weights.    
    def self.mutate(chromosome)
      if chromosome.normalized_fitness && rand < ((1 - chromosome.normalized_fitness) * 0.3)
        #TODO
        chromosome.data = data
        @fitness = nil
      end
    end
    
    # Reproduction here involves no crossover, simply duplication. 
    def self.reproduce(a, b)
      data_size = @@costs[0].length
      available = []
      0.upto(data_size-1) { |n| available << n }
      token = a.data[0]
      spawn = [token]
      available.delete(token)
      while available.length > 0 do 
        #Add to spawn
        token = next_token
        available.delete(token)
        spawn << next_token
      end
      return WeightChromosome.new(spawn)
    end
    
    # Initializes an individual solution (chromosome) for the initial 
    # population. Usually the chromosome is generated randomly, but you can 
    # use some problem domain knowledge, to generate better initial solutions.
    def self.seed
      # TODO
    end

    def self.set_cost_matrix(costs)
      @@costs = costs
    end
  end
end

