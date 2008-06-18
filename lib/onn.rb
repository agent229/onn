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
    def initialize(connections, states)
      @neurons = []
      @connections = connections

      # Iterate through the rows of states and create a new neuron for each state
      states.size1 times do |neuron_index|
        @neurons << OscillatorNeuron.new(states[neuron_index]) 
      end
      @neurons.reverse!
    end

    # Evaluates the input. When there are n input nodes, the input_state is an nx3 matrix, where the rows
    # correspond to nodes and the columns to amplitude, frequency, and phase (in that order).
    def eval(input_state)

      # Present input, propagate from input nodes
      input_state.size1 times do |input_index|
        @neurons[input_index].propagate(input_state[input_index])
      end
      # TODO how will propagate know which nodes we are talking about. should neurons know their connections?...?

      #Propagate through the rest
      output = []
      # TODO write propagation using connections matrix... 
      return output

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
      
      # Return net error by propagating thru with result of GA
    end
    
  end
  
  
  class OscillatorNeuron
    
    attr_accessor :error
    attr_accessor :expected_output
    attr_accessor :last_state
    attr_accessor :state
   
   # Initializes a new neuron with the given initial_state, which should be a Vector containing
   # in order, [amplitude, frequency, phase]
    def initialize(initial_state)
      # Instance state
      @error = nil
      @last_state = nil 
      @state = initial_state 
    end
   
    # TODO figure out what this is?
    def push(x)
      @pushed += x
    end
    
    def propagate(input = nil)
      # TODO write propagation!
    end
    
    def calc_error
      # TODO write error calculation
    end
    
    def change_weights
      # TODO write change weights (also depends on the type of evolution...)
    end

  end
  
end
