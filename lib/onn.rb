module OscillatorNeuralNetwork
  
  # This is an implementation of oscillator neural networks which uses a genetic
  # algorithm as the learning rule. It can learn by adjusting either connection weights
  # or the "natural" frequencies/amplitudes/phases of the oscillators, depending on the
  # type of genetic algorithm selected from the OscillatorGeneticAlgorithm module.
  #
  # This is based on the neural network from the ai4r library.
 
  require("rbgsl") # For Matrix structures

  class GeneticAlgorithmONN 

    # Initializes an ONN of n elements with any structure and simple oscillators. 
    #   connections: an nxn weighted connections Matrix (range: 0-1) where the ijth weight corresponds
    #                to a weighted connection from node i to node j
    #   states: an nx3 Matrix of the natural amplitudes, frequencies, and phases  of the neurons, 
    #           where the row index identifies the node and the column an attribute
    def initialize(connections, hidden_states, input_states, output_states)
      @connections = connections.duplicate
      @hidden_states = hidden_states.duplicate
      @input_states = input_states.duplicate
      @output_states = output_states.duplicate
    end

    # Evaluates the input. When there are n input nodes, the input_state is an nx3 matrix, where the rows
    # correspond to nodes and the columns to amplitude, frequency, and phase (in that order).
    def eval(input_state)
      # Set new input states
      @input_states = input_state.duplicate

      # Propagate through the rest
      # TODO write propagation using connections matrix... 
      return @output

    end

    # This method trains the network using the genetic algorithm.
    # 
    # input: Networks input (nx3 matrix form)
    # output: Expected output for the given input (nx3 matrix form).
    #
    # This method returns the network error (not an absolut amount, 
    # the difference between real output and the expected output)
    def train(input, output)
      # This entire thing just needs to call the genetic algorithm for training..... TODO
      # GA then calls the eval method to obtain its results, weight the fitness, etc. 
      # Return net error by propagating thru with result of GA
      # Return of GA will be the network.
      # Different train methods?? (trainWeights, trainAmps....) or is it possible to just modify that some other way
    end
    
  end

end
