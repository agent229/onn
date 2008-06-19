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
    #   input_states: a Matrix of the natural properties of the input nodes 
    #   hidden_states: a Matrix of the natural properties of the hidden nodes 
    #   output_states: a Matrix of the natural properties of the output nodes 
    def initialize(connections, input_states, hidden_states, output_states)
    
      # Copy input data into the private fields
      @connections = connections.duplicate
      @hidden_states = hidden_states.duplicate
      @input_states = input_states.duplicate
      @output_states = output_states.duplicate

      # Store the number of nodes in each category
      @num_inputs = @input_states.size1
      @num_hidden = @hidden_states.size1
      @num_outputs = @output_states.size1
      @num_nodes = @num_inputs + @num_hidden + @num_outputs

    end

    # Evaluates new input, given in matrix form. 
    #   input_state: a new input in matrix form.
    def eval(input_state)

      # Set new input states
      @input_states = input_state.duplicate

      # Traverse connections matrix
      0.upto(@num_nodes-1) do |driver|
        0.upto(@num_nodes-1) do |reciever|
          if(reciever!=0.0)
           # TODO drive the signal through to recievers.... need to figure out, how do i do this? 
          end
        end
      end

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
      chrom_size = @connections.size1
      seed = 1
      ga = GeneticSearch.new(self, pop_size, gens, chrom_size, seed)
      self = ga.run 
      eval(@input_states)
    end
    
  end

end
