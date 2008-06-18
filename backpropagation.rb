
# The utility of artificial neural network 
# models lies in the fact that they can be used 
# to infer a function from observations. 
# This is particularly useful in applications 
# where the complexity of the data or task makes the 
# design of such a function by hand impractical.
# Neural Networks are being used in many businesses and applications. Their 
# ability to learn by example makes them attractive in environments where 
# the business rules are either not well defined or are hard to enumerate and 
# define. Many people believe that Neural Networks can only solve toy problems.
# Give them a try, and let you decide if they are good enough to solve your 
# needs.
# 
# In this module you will find an implementation of neural networks
# using the Backpropagation is a supervised learning technique (described 
# by Paul Werbos in 1974, and further developed by David E. 
# Rumelhart, Geoffrey E. Hinton and Ronald J. Williams in 1986)
# 
# More about neural networks and backpropagation:
# 
# * http://en.wikipedia.org/wiki/Backpropagation
# * http://en.wikipedia.org/wiki/Neural_networks
# 
# Author::    Sergio Fierens
# License::   MPL 1.1
# Project::   ai4r
# Url::       http://ai4r.rubyforge.org/
#
# Specials thanks to John Miller, for several bugs fixes and comments in the
# Backpropagation implementation
#
# You can redistribute it and/or modify it under the terms of 
# the Mozilla Public License version 1.1  as published by the 
# Mozilla Foundation at http://www.mozilla.org/MPL/MPL-1.1.txt
# 

module NeuralNetwork
  
  # = Introduction
  # 
  # This is an implementation of neural networks
  # using the Backpropagation is a supervised learning technique (described 
  # by Paul Werbos in 1974, and further developed by David E. 
  # Rumelhart, Geoffrey E. Hinton and Ronald J. Williams in 1986)
  # 
  # = How to use it
  # 
  #   # Create the network
  #   net = Backpropagation.new([4, 3, 2])  # 4 inputs
  #                                         # 1 hidden layer with 3 neurons, 
  #                                         # 2 outputs
  #   # Train the network 
  #   1..upto(100) do |i|
  #     net.train(example[i], result[i])
  #   end
  #   
  #   # Use it: Evaluate data with the trained network
  #   net.eval([12, 48, 12, 25])  # =>  [0.86, 0.01]
  #   
  class Backpropagation

    DEFAULT_BETA = 0.5
    DEFAULT_LAMBDA = 0.25
    DEFAULT_THRESHOLD = 0.66
    
  # Creates a new network specifying the its architecture.
  # E.g.
  #    
  #   net = Backpropagation.new([4, 3, 2])  # 4 inputs
  #                                         # 1 hidden layer with 3 neurons, 
  #                                         # 2 outputs    
  #   net = Backpropagation.new([2, 3, 3, 4])   # 2 inputs
  #                                             # 2 hidden layer with 3 neurons each, 
  #                                             # 4 outputs    
  #   net = Backpropagation.new([2, 1])   # 2 inputs
  #                                       # No hidden layer
  #                                       # 1 output
  #
  # Optionally you can customize certain parameters:
  # 
  # threshold = A real number which we will call Threshold. 
  # Experiments have shown that best values for q are between 0.25 and 1. 
  # 
  # lambda = The Learning Rate: a real number, usually between 0.05 and 0.25.
  # 
  # momentum = A momentum will avoid oscillations during learning, converging 
  # to a solution in less iterations.
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
  #     net = Backpropagation.new([4, 3, 2])
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
  
  
  class Neuron
    
    attr_accessor :state
    attr_accessor :error
    attr_accessor :expected_output
    attr_accessor :w
    attr_accessor :x
    
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