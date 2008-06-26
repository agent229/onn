# This is an implementation of oscillator neural networks which uses a genetic
# algorithm as the learning rule. It learns by adjusting either individual connection weights
# or the "natural" properties of the oscillators, depending on the need of the user.
#
# Based on the neural network from the ai4r library.

module OscillatorNeuralNetwork
 
  # For genetic algorithm
  require 'onn_genetic_algorithm'
  import ONNGeneticAlgorithm      

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
      # Seed PRNG for this run
      srand(seed)
    end

    # Stores input data array in nodes as list.
    def set_nodes(state_vals)
      state_vals.each { |nstate| @nodes << OscillatorNeuron.new(Hash.create(@state_names, nstate)) } 
      # Put nodes back in correct index order
      @nodes.reverse!
    end

    # Generates random set of nodes (random natural states)
    def generate_random_nodes
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
    # Stores the connection information in each node's in_conns and out_conns fields.
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

    # Evaluates the network's state by propagating through a new input, given in 2D array form. 
    #   input_state: a new input (2D array format)
    #   num_outputs: the number of output nodes in the network
    def eval(input_state, num_outputs)
      # Set new input states
      change_input(input_state)
      # Tell every node to propagate its current state to its out_conns
      @nodes.each do |node|
        node.propagate
      end
      # After propagation, calculate and update each node's state 
      @nodes.each do |node|
        node.update_state
      end
      # Return the calculated output as a 2D array of output nodes
      (@nodes.length-num_outputs).upto(@nodes.length) do |node|
        output << node
      end
      output.reverse!
      return output
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
    #   output: Expected output for the given input (nested/2D array form)
    #   pop_size: how many solutions to evolve in the population
    #   gens: how many generations to run the GA
    #   seed: a PRNG seed governing all PRNG uses in this run (for repeatability)
    #
    # Returns: the difference between real output and the expected output
    def train(input, output, pop_size, gens)
      # Set new input
      change_input(input)
      # Instantiate genetic algorithm
      ga = GeneticSearch.new(self, pop_size, gens)
      # Run genetic algorithm
      @nodes = ga.run(modify_rules)
      # Evaluate the result
      output = eval(input, output.length)
      # Calculate and return a weighted error
      return weighted_error(output)
    end

    # Error weighting function: will take certain parameters and calculate a weighted error
    def weighted_error(result)
      # TODO define weighted error
    end

  end

  # This class describes a single OscillatorNeuron. Each neuron knows everything about
  # its own natural state, current state, next state, and inbound/outbound connections.
  class OscillatorNeuron

    # A list of existing OscillatorNeuron objects which point here hashed with conn weights
    attr_accessor :in_conns
    # A list of existing OscillatorNeuron objects which this points at hashed with conn weights
    attr_accessor :out_conns 
    # State is a hash, containing all information (besides connections)
    attr_accessor :curr_state
    # State is a hash, containing all information (besides connections)
    attr_accessor :next_state

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

    # Updates the current state based on the inputs to this node. 
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
