# GENERAL TODO: modify everything to use the new node objects, hashes, etc. create methods to automate the conversions
# from matrices to these objects and hashes

module OscillatorNeuralNetwork
  
  # This is an implementation of oscillator neural networks which uses a genetic
  # algorithm as the learning rule. It can learn by adjusting either connection weights
  # or the "natural" properties of the oscillators, depending on the
  # type of genetic algorithm class selected from the OscillatorGeneticAlgorithm module.
  #
  # The general network data structures are Matrices as defined in Ruby/GSL. The rows always
  # correspond to a particular node, and columns to attributes or data associated with that node.
  #
  # This is based on the neural network from the ai4r library.
 
  require("rbgsl")                       # For Matrix structures
  require 'oscillator_genetic_algorithm' # For genetic algorithm classes
  import OscillatorGeneticAlgorithm      # For genetic algorithm classes

  class GeneticAlgorithmONN 

    # Initializes an ONN of n nodes with any structure and coupled universal oscillators. 
    #   connections: an nxn, weighted connections Matrix (entry range: 0-1) where the ijth weight corresponds
    #                to a weighted, asymmetric connection from node i to node j
    #   states: a Matrix of the natural properties of the nodes, ordered in the same way as the
    #           connections matrix, with the same row indices, data in columns
    def initialize(connections, states)
    
      # Create the network, which is going to be an indexed list of OscillatorNeuron objects
      @nodes = []
      0.upto(states.size1) do |index|
        # TODO create nstate hash from the states matrix somehow
        @nodes << OscillatorNeuron.new(index, nstate) 
      end

      # Now that the network is created, go through and update the connections TODO
      0.upto(connections.size1) do |row|
        0.upto(connections.size1) do |col|
          if connections[row,col] != 0
            # TODO convert into connection, store in appropriate node
          end
        end
     end

    end

    # Evaluates new input, given in matrix form. 
    #   input_state: a new input in matrix form.
    def eval(input_state)

      # Set new input states
      0.upto(input_state.size1) do |index|
        @nodes[index].state = #TODO make a "create hash from state data" method to call here and above 
      end

      # Traverse network and drive oscillations and calculate output TODO 

      # Return the calculated output
      return @output

    end

    # This method trains the network using an instance of an OscillatorGeneticAlgorithm.
    #   input: Networks input (matrix form)
    #   output: Expected output for the given input (matrix form).
    #
    # Returns: the difference between real output and the expected output
    def train(input, output)
      # This entire thing just needs to call the genetic algorithm for training..... TODO
      # GA then calls the eval method to obtain its results, weight the fitness, etc. 
      # Return net error by propagating thru with result of GA
      # Return of GA will be the network.
      # Different train methods?? (trainWeights, trainAmps....) or is it possible to just modify that some other way
      pop_size = 10;
      gens = 10;
      seed = 1
      ga = GeneticSearch.new(@nodes, pop_size, gens, seed, params)
      @nodes = ga.run
      eval(@input_states)
    end
    
  end

  # This class keeps track of the state of a single OscillatorNeuron. Each neuron knows everything about
  # its own current state and connections (both inbound and outbound connections).
  class OscillatorNeuron

    attr_accessor :in_conns # A list of exiting OscillatorNeuron objects which point here hashed with conn weights
    attr_accessor :out_conns # A list of exiting OscillatorNeuron objects which this points at hashed with conn weights
    attr_accessor :state # State will likely be a hash, containing all variables and a type (input, hidden, output)?

    # Initialize a new OscillatorNeuron by passing a state hash.
    #  state: a has describing all of the state variables (TBD contents.... TODO)
    #  index: the node's original index from the connections/state matrices
    def initialize(index,state)
      @index = index
      @state = state
      @in_conns = Hash.new 
      @out_conns = Hash.new 
    end

    # Instantiates the list of inbound connections
    def set_in_conns(in_conns)
      #TODO FIX
      @in_conns = in_conns
    end

    # Adds an outbound connection to the list
    def add_out_conn(out_conn)
      # TODO FIXME to use hash
      @out_conns << out_conn
    end

  end

end
