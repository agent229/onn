module OscillatorNeuralNetwork
  
  # This is an implementation of oscillator neural networks which uses a genetic
  # algorithm as the learning rule. It can learn by adjusting either connection weights
  # or the "natural" frequencies/amplitudes/phases of the oscillators, depending on the
  # type of genetic algorithm selected from the OscillatorGeneticAlgorithm module.
  # This is based on the neural network from the ai4r library.
  
  class GeneticAlgorithmONN 

    # Initializes the ONN with the given connections matrix and frequencies, phases, and
    # amplitudes vectors.
    def initialize(connections, frequencies, phases, amplitudes)
      @neurons = []
      layer_sizes.reverse.each do |layer_size|
        layer = []
        layer_size.times { layer <<  Neuron.new(@neurons.last, threshold, lambda, momentum) }
        @neurons << layer
      end
      @neurons.reverse!
    end

    # Evaluates the input, which should sets of frequencies/phases/amplitudes, one set
    # for each input node.
    def eval(input)
      #check input size

      #Present input
      input.each_index do |input_index|
        @neurons.first[input_index].propagate(input[input_index])
      end
      #Propagate
      @neurons[1..-1].each do |layer|
        layer.each {|neuron| neuron.propagate}
      end
      output = []
      @neurons.last.each { |neuron| output << neuron.state }
      return output
    end

    # This method trains the network using the genetic algorithm.
    # 
    # input: Networks input
    # output: Expected output for the given input.
    #
    # This method returns the network error (not an absolut amount, 
    # the difference between real output and the expected output)
    def train(input, output)
      #Eval input
      eval(input)
      #Set expected output
      output.each_index do |output_index|
        @neurons.last[output_index].expected_output = output[output_index]
      end
      #Calculate error
      @neurons.reverse.each do |layer|
        layer.each {|neuron| neuron.calc_error}
      end
      #Change weight
      @neurons.each do |layer|
        layer.each {|neuron| neuron.change_weights }
      end
      #return net error
      return @neurons.last.collect { |x| x.calc_error }
    end

    private
    def print_weight
      @neurons.each_index do |layer_index|
        @neurons[layer_index].each_index do |neuron_index| 
          puts "L #{layer_index} N #{neuron_index} W #{@neurons[layer_index][neuron_index].w.inspect}"
        end
      end
    end
    
  end
  
  
  class OscillatorNeuron
    
    attr_accessor :state
    attr_accessor :error
    attr_accessor :expected_output
    attr_accessor :w
    attr_accessor :x
    attr_accessor :amp
    attr_accessor :phase
    attr_accessor :freq
    
    def initialize(childs, threshold, lambda, momentum)
      #instance state
      @w = nil
      @childs = childs
      @error = nil
      @state = 0
      @pushed = 0
      @last_delta = 0
      @x = 0
      @last_amp = amp
      @last_phase = phase
      @last_freq = freq

      #Parameters
      @lambda = lambda
      @momentum = momentum
      @threshold = threshold
      @amp = amp
      @phase = phase
      @freq = freq

      #init w     
      if(childs)
        @w = []
        childs.each { @w << init_weight }
      end
    end
    
    def push(x)
      @pushed += x
    end
    
    def propagate(input = nil)

    end
    
    def calc_error
      if(!@childs && @expected_output)
        @error = (@expected_output - @state)
      elsif(@childs)
        @error = 0
        @childs.each_index do |child_index|
          @error += (@childs[child_index].error * @w[child_index])
        end
      end
    end
    
    def change_weights

    end
    
    private 
    def init_weight
      rand/4
    end

  end
  
end
