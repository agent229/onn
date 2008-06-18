module OscillatorNeuralNetwork
  
  # = Introduction
  # 
  # This is an implementation of oscillatory neural networks which uses a genetic
  # algorithm as the learning rule. It can learn by adjusting connection weights
  # or the "natural" frequencies/amplitudes/phases of the oscillators, depending on the
  # type of genetic algorithm selected from the OscillatorGeneticAlgorithm module.
  # 
  # = How to use it
  # 
  #   
  class GeneticAlgorithmONN 

    def initialize(layer_sizes, threshold=DEFAULT_THRESHOLD, lambda=DEFAULT_LAMBDA, momentum=DEFAULT_BETA)
      @neurons = []
      layer_sizes.reverse.each do |layer_size|
        layer = []
        layer_size.times { layer <<  Neuron.new(@neurons.last, threshold, lambda, momentum) }
        @neurons << layer
      end
      @neurons.reverse!
    end

  # Evaluates the input.
  # E.g.
  #     net = GeneticAlgorithmONN.new([4, 3, 2])
  #     net.eval([25, 32.3, 12.8, 1.5])
  #         # =>  [0.83, 0.03]
    def eval(input)
      #check input size
      if(input.length != @neurons.first.length)
        raise "Wrong input dimension. Expected: #{@neurons.first.length}" 
      end
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

    # This method trains the network using the backpropagation algorithm.
    # 
    # input: Networks input
    # 
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
      if(input)
        input = input.to_f
        @x = input
        @state = input
        @childs.each_index do |child_index| 
          @childs[child_index].push(input * @w[child_index])
        end
      else
        @x = @pushed + @threshold
        @pushed = 0
        @state = Neuron.f(@x)
        if @childs
          @childs.each_index do |child_index| 
            @childs[child_index].push(@state * @w[child_index])
          end
        end
      end
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
      return if !@childs
      @childs.each_index do |child_index |
        delta = @lambda * @childs[child_index].error * (@state) * Neuron.f_prime(@childs[child_index].x)
        @w[child_index] += (delta + @momentum * @last_delta)
        @last_delta = delta
      end
    end
  
    # Propagation function.
    # By default: 
    #   f(x) = 1/(1 + e^(-x))
    # You can override it with any derivable function.
    # A usually usefull one is: 
    #   f(x) = x.
    # If you override this function, you will have to override
    # f_prime too.
    def self.f(x)
      return 1/(1+Math.exp(-1*(x)))
    end
    
    # Derived function of the propagation function (self.f)
    # By default: 
    #   f_prime(x) = f(x)(1- f(x))
    # If you override f(x) with:
    #   f(x) = x.
    # Then you must override f_prime as:
    #   f_prime(x) = 1
    def self.f_prime(x)
      val = f(x)
      return val*(1-val)
    end
    
    private 
    def init_weight
      rand/4
    end

  end
  
end
