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

  # "Genetic algorithm oscillator neural network" (GAONN) class definition
  class GAOscillatorNeuralNetwork 
   
    # Access to the array of OscillatorNeuron objects comprising the network
    attr_accessor :nodes
    # Ready-only access to other fields (for testing/data output)
    attr_reader :t_step
    attr_reader :num_nodes
    attr_reader :connections

    # Initializes an ONN of coupled sinusoidal oscillators. (simple version)
    #   t_step: discrete time step to be used in simulation to update oscillators
    #   num_nodes: the number of nodes to create in the network
    #   seed: a PRNG seed governing all PRNG uses in this run (for repeatability)
    #   connections: connections matrix
    def initialize(t_step, num_nodes, seed, connections)
      # Initialize some instance variables
      @t_step = t_step
      @nodes = []
      @num_nodes = num_nodes
      @connections = connections
      # TODO un-hard-code names?
      @state_names = [:natural_freq, :natural_phase]
      # Seed PRNG for this run
      srand(seed)
    end

    # Stores input data array as list of OscillatorNeuron objects instantiated with data.
    def set_nodes(state_vals)
      state_vals.each { |nstate| @nodes << OscillatorNeuron.new(Hash.create(@state_names, nstate)) } 
    end

    # Generates 'random' set of node data (random natural states)
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

    # Generates random connections matrix (TODO deal with inputs/outputs restrictions)
    def generate_random_connections
      conns = []
      @num_nodes.times do
        row = []
        @num_nodes.times do
          row << rand 
        end
        conns << row
      end
      return conns
    end

    # Updates the connections between nodes based on the connections 2D array/matrix.
    # Stores the connection information in the approproiate node's in_conns and out_conns fields.
    #  connections: the 2D (nested) array of weighted connections
    def update_connections(connections)
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

    # Evaluates the network's state by propagating through the current input states. To
    # propagate a enw input state, first call change_input
    def eval
      # Tell every node to propagate its current state to its out_conns
      @nodes.each do |node|
        node.propagate
      end
      # After propagation, calculate and update each node's state 
      @nodes.each do |node|
        node.update_state
      end
      # Return the calculated output as a 2D array of output nodes
      return get_outputs
    end

    # Retrieves all output nodes by checking which have no out_conns 
    def get_outputs
      count = 0
      @nodes.each do |node|
        if node.out_conns.empty?
          outputs << node
        end 
      end
      return outputs # TODO is order preserved? is order needed?
    end

    # Resets the input states manually
    #  new_input: a 2D array describing the desired state of the input nodes
    def change_input(new_input)
      new_input.each_index do |index| 
        @nodes[index].curr_state = Hash.create(@state_names,new_input[index]) 
      end 
    end

    # This method trains the network using an instance of an OscillatorGeneticAlgorithm.
    #   input: Network input (nested/2D array form)
    #   output: Expected/desired output for the given input (nested/2D array form)
    #   pop_size: how many solutions to evolve in the population
    #   gens: how many generations to run the GA
    #
    # Returns: the difference between real output and the expected output
    def train(input, exp_output, pop_size, gens)
      # new instance vars for current training run
      @curr_expected = exp_output
      # Set new input
      change_input(input)
      # Instantiate genetic algorithm
      ga = GeneticSearch.new(self, pop_size, gens)
      # Run genetic algorithm, getting the set of nodes back
      @nodes = ga.run
      # Evaluate the result with the new, GA-modified nodes in place
      actual_output = eval
      # Calculate and return a weighted error
      err = weighted_error(actual_output, exp_output)
      @curr_expected = nil
      return err
    end

    # Error weighting function. calculates a weighted error measure of the output
    #   result: the actual result of a network propagation
    #   expected: the expected/desired result
    def weighted_error(result, expected)
      result.each_index do |rows|
        result[rows].each_index do |columns|
          # TODO cleanup/fix/figure out
          w_err += Math::abs(result[columns][rows]-expected[columns][rows])
        end
      end
      return w_err
    end

    # Fitness function for GA. Returns a normalized fitness value (0-1)
    def fitness
     output = self.eval  
     err = weighted_error(output,@curr_expected)
     return err
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
      @next_state[:phase] += @input_sum_terms.inject(0){ |sum,item| sum + item } + @curr_state[:freq]
      @curr_state = @next_state
      @next_state = nil
    end

    # Resets the current state to the "natural" state
    def reset_state
      @curr_state = @natural_state
    end

    # Propagates the node's curr_state to all of its out_conns
    def propagate
      # Iterate through all outgoing connections
      @out_conns.each_key do |receiver|
        # Calculate the term of the sum corresponding to the propagating node
        term = @out_conns[receiver] * Math::sin(@curr_state['phase']-receiver.curr_state['phase'])
        # Insert the term in the receiver's registry
        receiver.input_sum_terms << term
      end
    end

  end

end
