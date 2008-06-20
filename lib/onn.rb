module OscillatorNeuralNetwork
  
  # This is an implementation of oscillator neural networks which uses a genetic
  # algorithm as the learning rule. It can learn by adjusting either connection weights
  # or the "natural" properties of the oscillators, depending on the need of the user.
  #
  # The general network data structures are input as 2D (nested) arrays. The rows always
  # correspond to a particular node, and columns to attributes or data associated with that node.
  #
  # This is based on the neural network from the ai4r library.
 
  require 'oscillator_genetic_algorithm' # For genetic algorithm classes
  import OscillatorGeneticAlgorithm      

  # Custom Hash method to convert an array of keys and an array of values to a Hash
  #  keys: an array containing the ordered keys
  #  values: an array containing the ordered values
  # Usage:
  #   Hash.create(['Oriole','Spaniel'],['bird','dog']) => {"Oriole"=>"bird", "Spaniel"=>"dog"}
  class << Hash
    def create(keys, values)
      self[*keys.zip(values).flatten]
    end
  end

  # Network class definition
  class GeneticAlgorithmONN 

    attr_accessor :nodes

    # Initializes an ONN of n nodes with any structure and coupled universal oscillators. 
    #   connections: an nxn, weighted 2D connections array (entry range: 0-1) where the ijth weight corresponds
    #                to a weighted, asymmetric connection from node i to node j
    #   state_names: an array of the names of the state variables provided in state_vals, in the same
    #                order as the columns in state_vals
    #   state_vals: a 2D array of the natural properties of the nodes, ordered in the same way as the
    #           connections array, with the same row indices, data in columns
    # Usage:
    #  net = GeneticAlgorithmONN.new([[0,0],[1,1]],['phase','freq'],[[0.2,0.3],[0.5,0.6]])
    def initialize(connections, state_names, state_vals)
    
      # Create the network, which is an indexed list of OscillatorNeuron objects
      @nodes = []
      @state_names = state_names
      state_vals.each { |nstate| @nodes << OscillatorNeuron.new(Hash.create(@state_names, nstate)) } 
      # Put nodes back in correct index order
      @nodes.reverse!

      # Now that the network is created, go through and update the connections 
      update_connections(connections)

    end

    # Updates the connections between nodes based on the connections matrix
    #  connections: the 2D (nested) array of weighted connections
    def update_connections(connections)
      connections.each_index { |row_index|
        connections[row_index].each_index { |col_index|
          this_conn = connections[row_index][col_index]
          if this_conn != 0.0
            @nodes[row_index].out_conns[@nodes[col_index]] = this_conn 
            @nodes[col_index].in_conns[@nodes[row_index]] = this_conn 
          end
        }
      } 

    end

    # Evaluates new input, given in 2D array form. 
    #   input_state: a new input.
    def eval(input_state)

      # Set new input states
      input_state.each_index { |index| @nodes[index].state = Hash.create(@state_names,input_state[index]) }

      # Traverse network and drive oscillations and calculate output TODO 

      # Return the calculated output
      return @output

    end

    # This method trains the network using an instance of an OscillatorGeneticAlgorithm.
    #   input: Networks input (nested/2D array form)
    #   output: Expected output for the given input (nested/2D array form).
    #   pop_size: how many solutions to keep in the population
    #   gens: how many generations to run the GA
    #   seed: a PRNG seed governing all PRNG uses in this run (for repeatability)
    #
    # Returns: the difference between real output and the expected output
    def train(input, output, pop_size, gens, seed, params)
      # This entire thing just needs to call the genetic algorithm for training..... TODO
      # Determine structure of params TODO
      ga = GeneticSearch.new(@nodes, pop_size, gens, seed, params)
      @nodes = ga.run
      eval(@input_states)
    end
    
  end

  # This class keeps track of the state of a single OscillatorNeuron. Each neuron knows everything about
  # its own natural state, current state, and inbound/outbound connections.
  class OscillatorNeuron

    attr_accessor :in_conns # A list of existing OscillatorNeuron objects which point here hashed with conn weights
    attr_accessor :out_conns # A list of existing OscillatorNeuron objects which this points at hashed with conn weights
    attr_accessor :state # State is a hash, containing all information (besides connections)

    # Initialize a new OscillatorNeuron by passing a state hash.
    #  state: a hash describing all of the state variables 
    def initialize(state)
      @state = state
      @in_conns = Hash.new 
      @out_conns = Hash.new 
    end

  end

end
