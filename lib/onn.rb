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
   
    attr_accessor :nodes
    attr_reader :t_step
    attr_reader :num_nodes
    attr_reader :seed

    # Initializes an ONN of coupled sinusoidal oscillators. 
    #   t_step: discrete time step to be used in simulation to update oscillators
    #           (smaller -> closer to continuous time)
    #   num_nodes: the number of nodes that will exist in the network
    #   seed: a PRNG seed governing all PRNG uses in this run (for repeatability)
    def initialize(t_step, num_nodes, seed)
      # Initialize instance variables from parameters
      @t_step = t_step
      @num_nodes = num_nodes
      @seed = seed
      
      # Create empty list of nodes
      @nodes = []

      # The array of names which describe modifiable features of the oscillators
      @state_names = [:freq, :phase]

      # Seed PRNG for this run
      srand(seed)
    end

    # Stores input data array as list of OscillatorNeuron objects instantiated with data.
    #   state_vals: a nested array of data describing OscillatorNeurons. Each entry is a list of
    #               properties in the order described by @state_names
    def set_nodes(state_vals)
      state_vals.each do |nstate| 
        @nodes << OscillatorNeuron.new(Hash.create(@state_names, nstate))
      end
    end

    # Generates a nested data array with random set of node data (random natural states).
    # Uses @num_nodes to determine how many nodes worth of data to generate.
    def generate_random_node_data
      data = []
      @num_nodes.times do
        row = []
        @state_names.each do
          row << rand 
        end
        data << row
      end
      return data
    end

    # Updates the connections between nodes based on a connections 2D array.
    # Stores the connection information in the approproiate node's in_conns and out_conns fields.
    #  connections: the 2D (nested) array of weighted connections, where the row, R, refers to the
    #               node the connection is coming from, and the column C, when nonzero, refers
    #               to the weighted connection to the Cth node
    def set_connections(connections)
     # Iterate through each entry in the 2D array
      connections.each_index do |row_index|
        connections[row_index].each_index do |col_index|
          this_conn = connections[row_index][col_index]
          if this_conn != 0.0 
            @nodes[row_index].out_conns[@nodes[col_index]] = this_conn 
            @nodes[col_index].in_conns[@nodes[row_index]] = this_conn 
          end
        end 
      end 
    end

    # Evaluates the network's state by propagating the current input states through the network.
    # Returns a list containing the output states.
    def eval
      # Tell every node to propagate/communicate its current state to its outgoing connections
      @nodes.each do |node|
        node.propagate
      end

      # After propagation, calculate/update each node's state 
      @nodes.each do |node|
        node.update_state
      end

      # Return the calculated output as a 2D array of output nodes
      return get_outputs
    end

    # Retrieves all output nodes by checking which have no outgoing connections (I define
    # this to be an output node). Returns a list of output nodes.
    def get_outputs
      outputs = []
      @nodes.each do |node|
        if node.out_conns.empty?
          outputs << node
        end 
      end
      return outputs 
    end

    # Replaces the input nodes with new nodes described in the given data array. 
    #  new_input: a 2D array describing the desired state of the input nodes, which should 
    #             be given in the same order as the original nodes
    def change_input(new_input)
      new_input.each_index do |index| 
        @nodes[index].curr_state = Hash.create(@state_names,new_input[index]) 
      end 
    end

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
    # TODO improve (very simple right now)
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

    # Convenience method that converts a list of node objects into a nested data array
    # representing them, preserving order. Returns the array.
    def nodes_to_data_arr(nodelist)
      data_arr = []
      nodelist.each do |node|
        node_data = []
        @state_names.each do |name|
          node_data << node.curr_state[name]
        end
        data_arr << node_data
      end
      return data_arr
    end

    # GA fitness function. Takes a nodelist and returns a weighted error calculation of the result
    # compared with the expected/desired result by evaluating the network.
    #   chromosome: a list of OscillatorNeurons 
    def fitness(chromosome)
     @nodes = chromosome
     output = eval  
     err = weighted_error(output,@curr_expected)
     return err
    end

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

    # A list of existing OscillatorNeuron objects which connect to this neuron, hashed with conn weights
    attr_accessor :in_conns
    # A list of existing OscillatorNeuron objects which this neuron connects to, hashed with conn weights
    attr_accessor :out_conns 
    # State is a hash, containing all information (besides connections)
    attr_accessor :curr_state
    # State is a hash, containing all information (besides connections)
    attr_accessor :next_state
    # State is a hash, containing all information (besides connections)
    attr_reader :natural_state
    attr_accessor :input_sum_terms

    # Initialize a new OscillatorNeuron by passing a "natural state" hash.
    #   natural_state: a hash describing all of the "natural" state variables, with names as keys 
    def initialize(natural_state)
      # Set states
      @curr_state = natural_state
      @natural_state = natural_state
      @next_state = nil
      # Reserve space for other instance variables
      @in_conns = Hash.new 
      @out_conns = Hash.new
      @input_sum_terms = []
    end

    # Updates the current state based on the current states of inputs to this node.  
    def update_state
      @next_state = @curr_state
      # Traditional Kuramoto-style update rule, with variable connection weight capability
      sum = @input_sum_terms.inject(0){|sum,item| sum+item}
      @next_state[:phase] += @natural_state[:freq] - sum
      @curr_state = @next_state
      @next_state = nil
    end

    # Resets the current state to the "natural" state
    def reset_state
      @curr_state = @natural_state
      @next_state = nil
    end

    # Propagates the node's curr_state to all of its out_conns
    def propagate
      # Iterate through all outgoing connections
      @out_conns.each_key do |receiver|
        # Calculate the term of the sum corresponding to the propagating node
        term = @out_conns[receiver] * Math::sin(@curr_state[:phase]-receiver.curr_state[:phase])
        # Insert the term in the receiver's registry
        receiver.input_sum_terms << term
      end
    end

  end

end
