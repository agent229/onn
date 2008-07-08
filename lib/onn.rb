# This is an implementation of oscillator neural networks which uses a genetic
# algorithm as the learning rule. It learns by adjusting either individual connection weights
# or the "natural" properties of the oscillators, depending on the need of the user.
#
# (Loosely) based on the neural network from the ai4r library.

module OscillatorNeuralNetwork
 
  # For genetic algorithm
  require '~/Documents/SFI/NN/onn/lib/onn_genetic_algorithm'
  include ONNGeneticAlgorithm      

  # Custom Hash convenience method: converts an array of keys and an array of values to a Hash
  #  keys: an array containing the ordered keys
  #  values: an array containing the ordered values
  # Usage:
  #   Hash.create(['Oriole','Spaniel'],['bird','dog']) => {"Oriole"=>"bird", "Spaniel"=>"dog"}
  class << Hash
    def create(keys, values)
      self[*keys.zip(values).flatten]
    end
  end

  # "Genetic algorithm oscillator neural network" (GAONN) class definition describes an
  # entire network object through a list of nodes
  class GAOscillatorNeuralNetwork 
  
    # A nested array where the rows are a layer, and each layer contains OscillatorNeurons which belong to it
    attr_accessor :layers
    # Parameters
    attr_reader :time_param
    attr_reader :window_param
    attr_reader :seed

    # Defaults
    DEFAULT_TIME_PARAM = 0.2
    DEFAULT_WINDOW_PARAM = 20
    DEFAULT_SEED = 0

    # Initializes an ONN of coupled sinusoidal oscillators. 
    #   node_data: a list of vectors containing the following data: <layer_number, natural_frequency, natural_amplitude>
    #          (same order as in the connections matrix for consistency)
    #   connections: a connections matrix (nested array) of connection strengths in the network where ijth entry corresponds
    #                to the asymmetric connection from node i to node j, where i and j are the indices within the nodes list
    #   seed: a PRNG seed governing all PRNG uses in this run (for repeatability)
    #   time_param: parameter to be used in simulation to help decide how often to update states 
    #   window_param: parameter to be used in simulation to help decide stability; related to a sampling "window size" 
    def initialize(node_data, connections, seed=DEFAULT_SEED, time_param=DEFAULT_TIME_PARAM, window_param=DEFAULT_WINDOW_PARAM)
      # Initialize instance parameters
      @time_param = time_param
      @window_param = window_param
      @seed = seed

      # Initialize network of nodes by layer
      @nodes = create_node_list(node_data)
      @connections = connections

      # Seed PRNG for this run
      srand(seed)
    end

    # Creates the nested list of layers containing node objects with the specified nodes in each layer
    def create_node_list(node_data)
      nodes = []
      node_data.each do |node_datum|
        node_layer = node_datum[0]
        if !nodes[node_layer]
          nodes[node_layer] = []
        end
        nodes[node_layer] << OscillatorNeuron.new(node_datum) 
      end
      return nodes
    end

    # Evaluates the network's state by propagating the current input states through the network.
    # Returns a list containing the output states.
    def eval
      # Tell every node to propagate/communicate its current state to its outgoing connections
      @nodes.each do |layer|
        layer.each do |node|
          node.propagate
        end
      end

      # After propagation, calculate/update each node's state 
      @nodes.each do |layer|
        layer.each do |node|
          node.update_state
        end
      end

      # Return the calculated output as a 2D array of output nodes
      return @nodes.last 
    end

    # TODO fix up with new changes
    # This method trains the network using an instance of an OscillatorGeneticAlgorithm.
    #   input: Network input (ordered, nested/2D array form)
    #   exp_output: Expected/desired output for the given input (ordered, nested/2D array form)
    #   pop_size: how many solutions to keep in the "potential solution" population
    #   gens: how many generations to run the GA
    #   mutation_rate: a parameter describing how frequently a given solution gets random mutations
    # Returns a weighted error estimate of how "close" the expected and actual outputs are after
    # training is complete.
    def train(input, exp_output, pop_size, gens, mutation_rate)
      @curr_expected = exp_output
      change_input(input) # Set new input states

      # Instantiate genetic algorithm instance
      ga = GeneticSearch.new(self, pop_size, gens, mutation_rate)
      # Run genetic algorithm, getting the best set of nodes back
      @nodes = ga.run

      # Evaluate the result with the new, GA-modified nodes in place
      actual_output = eval

      # Calculate and return a weighted error
      err = weighted_error(actual_output, exp_output)

      return err
    end

    # Error weighting function. Calculates a weighted error measure of the output.
    #   result: the actual result of a network propagation (list of output nodes)
    #   expected: the expected/desired result (2D array of data)
    # TODO improve/fix (very simple right now)
    def weighted_error(result, expected)
      w_err = 0
      result_arr = nodes_to_data_arr(result)
      result_arr.each_index do |node_index|
        result_arr[node_index].each_index do |state_index|
          w_err += Math::abs(result_arr[node_index][state_index]-expected[node_index][state_index])
        end
      end
      return w_err
    end
 
    # TODO fix with layers
    # GA fitness function. Takes a nodelist and returns a weighted error calculation of the result
    # compared with the expected/desired result by evaluating the network.
    #   chromosome: a list of OscillatorNeurons 
    def fitness(chromosome)
     @nodes = chromosome
     output = eval  
     err = weighted_error(output,@curr_expected)
     return err
    end

    # TODO fix with layers
    # Mutation function. Randomly mutates with a given chance specified by GA.
    #  chromosome: a nodelist (list of Oscillator Neurons)
    #  mutation_rate: parameter describing the rate of mutation (chance of random mutation)
    # Returns the mutated chromosome
    def mutate(chromosome, mutation_rate)
      chromosome.each do |node|
        node.natural_state.each_value do |val|
          if rand < mutation_rate
            val += (rand - 0.5)  
          end
        end
      end
      return chromosome
    end

  end

  # This class describes a single OscillatorNeuron. Each neuron knows everything about
  # its own natural state, current state, next state, and inbound/outbound connections.
  class OscillatorNeuron
    attr_accessor :x
    attr_accessor :x_prime
    attr_accessor :x_prime_prime
    attr_accessor :layer
    attr_accessor :natural_freq
    attr_accessor :amplitude
    # A list of existing OscillatorNeuron objects which connect to this neuron, hashed with conn weights
    attr_accessor :in_conns
    # A list of existing OscillatorNeuron objects which this neuron connects to, hashed with conn weights
    attr_accessor :out_conns 
    # State is a hash, containing all information (besides connections)
    attr_accessor :input_sum_terms

    # Initialize a new OscillatorNeuron by passing a "natural state" hash.
    #   natural_state: a list describing the "natural" frequency, amplitude, and layer as such: <layer,frequency,amplitude> 
    def initialize(natural_state)
      # Set states
      @natural_freq = natural_state[1]
      @amplitude = natural_state[2]
      @x, @x_prime, @x_prime_prime = 0

      # Reserve space for other instance variables
      @in_conns = Hash.new 
      @out_conns = Hash.new
      @input_sum_terms = []
    end

    # TODO fix
    # Updates the current state based on the current states of inputs to this node.  
    def update_state
      @next_state = @curr_state
      # Traditional Kuramoto-style update rule, with variable connection weight capability
      sum = @input_sum_terms.inject(0){|sum,item| sum+item}
      @next_state[:phase] += @natural_state[:freq] - sum
      @curr_state = @next_state
      @next_state = nil
    end
 
    # TODO fix
    # Resets the current state to the "natural" state
    def reset_state
      @curr_state = @natural_state
      @next_state = nil
    end

    # TODO fix
    # Propagates the node's curr_state to all of its out_conns
    def propagate
      # Iterate through all outgoing connections
      @out_conns.each_key do |receiver|
        # Calculate the term of the sum corresponding to the propagating node
        term = @out_conns[receiver] * (x - receiver.x)
        # Insert the term in the receiver's registry
        receiver.input_sum_terms << term
      end
    end

  end

end
