# This is an implementation of oscillator neural networks which uses a genetic
# algorithm as the learning rule. It learns by adjusting either individual connection weights
# or the "natural" properties of the oscillators, depending on the need of the user.
#
# (Loosely) based on the neural network from the ai4r library.

module OscillatorNeuralNetwork
 
  # For genetic algorithm
  # TODO fix hardcoded path if deploying on linux
  require '~/Documents/SFI/NN/onn/lib/onn_genetic_algorithm'
  include ONNGeneticAlgorithm      

  # For Ruby/GSL scientific library (vectors, matrices, graphing)
  require 'gsl'

  # "Genetic algorithm oscillator neural network" (GAONN) class definition describes an
  # entire network object through a list of nodes and some parameters
  class GAONN 
  
    # An array of OscillatorNeuron objects 
    attr_accessor :nodes

    # Update rule parameters
    attr_reader :time_param
    attr_reader :seed

    # Default/suggested settings for parameters
    DEFAULT_TIME_PARAM = 0.2
    DEFAULT_SEED = 0
    DEFAULT_MUTATION_RATE = 0.4

    #### Class method(s) ####

    # This method trains a network using an instance of an OscillatorGeneticAlgorithm.
    #
    #   network:       the initial network to train 
    #   input:         GSL::Matrix of inputs to the network 
    #   exp_output:    GSL::Matrix of expected/desired output for the given input 
    #   pop_size:      How many solutions to keep in the "potential solution" population
    #   gens:          How many generations to run the GA
    #   mutation_rate: a parameter describing how frequently a given solution gets random mutations
    #
    # Returns an array containing the evolved network  along with a weighted error estimate of 
    # how "close" the expected and actual outputs are after training is complete.
    #
    def self.train(network, input, exp_output, pop_size, gens, mutation_rate=DEFAULT_MUTATION_RATE)
      change_input(input)                                            # Set new input states
      ga = GeneticSearch.new(network, pop_size, gens, mutation_rate) # Create GA
      best_net = ga.run                                              # Run GA
      # TODO determine exactly what gets returned from GA, what is neccessary to run it for error
      # TODO HERE, do the loop for evaluation... perhaps abstract somewhere to use other places too
      # (eval only evaluates once, decide here how many times to do this, for how long, fourier etc)
      actual_output = eval                                           # Evaluate trained network and get error
      err = weighted_error(actual_output, exp_output)
      return [best_net, err]
    end

    #### Network instance methods ####

    # Initializes an ONN of coupled harmonic oscillators. 
    #
    #   node_data:    a GSL::Matrix containing the following row vectors: <layer_number, natural_frequency, initial_amplitude>
    #   connections:  a GSL::Matrix of connection strengths in the network. ijth entry is connection from node i to node j
    #   seed:         a PRNG seed governing all PRNG uses in this run (for repeatability)
    #   time_param:   parameter to be used in simulation to help decide how often to update states (scales the minimum period)
    #
    def initialize(node_data, connections, seed=DEFAULT_SEED, time_param=DEFAULT_TIME_PARAM)
      @time_param = time_param                          # Set parameters
      @seed = seed
      @nodes = create_node_list(node_data, connections) # Initialize network of nodes by layer
      @connections = connections                        # Store connections GSL::Matrix
      @curr_time = 0.0                                  # Set current time to 0
      srand(seed)                                       # Seed PRNG for this run
      calc_time_step                                    # Calculate an appropriate time step
    end

    # Creates the list of OscillatorNeuron objects which contain the data 
    #
    #   node_data:   GSL::Matrix with rows containing the following data: <layer_number, natural_frequency, initial_amplitude>
    #   connections: GSL::Matrix of connection strengths
    #
    # Returns the list of nodes.
    def create_node_list(node_data, connections)
      nodes = []
      node_data.each_row do |node_datum|
        nodes << OscillatorNeuron.new(node_datum,self)    # Initialize node states
      end
      nodes = set_conns_from_mat(nodes, connections) # Set connections
      return nodes
    end

    # Evaluates the network's output state by propagating the current input states through the network.
    # Evaluates over one time step and returns the current set of outputs.
    def eval

      # Tell every node to propagate/communicate its current state to its outgoing connections
      @nodes.each do |node|
        node.propagate
      end

      # After propagation, calculate/update each node's state 
      @nodes.each do |node|
        node.update_state
      end

      increment_time

      # Store this set of output states for stability analysis
      @output_states << get_outputs

      # Return the calculated output as an array/list of output nodes
      return @output_states.last
    end

    # Error weighting function. Calculates a weighted error measure of the output.
    #   result: the actual result of a network propagation (list of output nodes)
    #   expected: the expected/desired result (2D array of data)
    def weighted_error(result, expected)
      w_err = 0
      result.each_index do |node_index|
        #TODO write function that estimates frequency, amp based on other data, use that here
        w_err += GSL::hypot(result[node_index].get_a-expected[node_index][0], result[node_index].freq-expected[node_index][0])
      end
      w_err = w_err / result.length
      return w_err
    end

  #### GA related functions ####

    # TODO fix with layers, make @curr_expected cleaner......
    # GA fitness function. Takes a nodelist and returns a weighted error calculation of the result
    # compared with the expected/desired result by evaluating the network.
    #   chromosome: a list of OscillatorNeurons 
    def fitness(chromosome)
      @nodes = chromosome
      output = eval  
      err = weighted_error(output,@curr_expected)
      return err
    end

    # Mutation function. Randomly mutates with a given chance specified by GA.
    #  chromosome: a nodelist (list of Oscillator Neurons divided into layers)
    #  mutation_rate: parameter describing the rate of mutation (chance of random mutation)
    # Returns the mutated chromosome
    def mutate(chromosome, mutation_rate)
      chromosome.each do |layer|
        layer.each do |node|
          # Add random mutations with chance mutation_rate
          if rand < mutation_rate
            node.natural_freq += (1/(2*GSL::M_PI) * (rand - 0.5)) % (1/(2*GSL::M_PI)) 
          end
        end
      end
      return chromosome
    end

  #### Helper/convenience functions ####

    # Calculates a good guess of a time step to use based on the minimum a (spring constant)
    # Returns a guess as to a good time step.
    def calc_time_step
      min_a = @nodes[0].get_a
      @nodes.each do |node|
        if(node.get_a < min_a)
          min_a = node.get_a
        end
      end
      return (min_a * @time_param)
    end

    # Stores connection information from the connections matrix into the nodes.
    #
    #   nodes:       list of OscillatorNeuron objects in order
    #   connections: connections GSL::Matrix describing connections between nodes
    #
    # Returns the nodelist with connections set
    def set_conns_from_mat(nodes, connections)
      pointer_index = 1
      reciever_index = 1
      connections.each_row do |pointer|
        pointer.each do |reciever|
          if receiver != 0.0
            nodes[receiver_index].in_conns = Hash.new(nodes[pointer_index], receiver)
          end
        reciever_index += 1
        end
        reciever_index = 1
        pointer_index += 1
      end
      return nodes
    end

    # Retreives the current output states as a GSL::Matrix of data (ordered)
    def get_outputs
      start_index = @nodes.length - @num_outputs
      @num_outputs.times do |output|

      end
      outputs = GSL::Matrix[@nodes[start_index]
    end

    # Sets the input nodes to different oscillator data
    #   new_input_data: a list of vectors describing the natural frequency and amplitudes of the inputs like <freq, amp>
    def change_input(new_input_data)
      @nodes[0].each_index do |node_index|
        @nodes[0][node_index].set_state(new_input_data[node_index])
      end
    end

    # Uses wavelet transform to get dominant frequency
    #  data_arr: an array of data points 
    def get_frequency(data_arr)
      #TODO fix up 
      vector = data_arr.to_gv
      # TODO run fourier analysis on vector
    end

    def get_time_step
      return @t_step
    end

    def get_current_time
      return @curr_time
    end

    def increment_time
      @curr_time += @t_step
    end

  end

  # This class describes a single OscillatorNeuron. Each neuron knows everything about
  # its own natural state, current state, next state, and inbound/outbound connections.
  # Variable state_vector is a GSL::vector containing the following information:
  # <a, b, x, x_prime, x_dbl_prime, layer> where the equation describing the
  # oscillator is x_dbl_prime = -a*x - b*x_prime + input_sum
  # so that a = spring constant, b = damping coefficient

  class OscillatorNeuron

    attr_accessor :state_vector
    attr_accessor :input_sum_terms
    attr_accessor :out_conns

    # Initialize a new OscillatorNeuron by passing a "natural state" hash.
    #   state_vec: a GSL::Vector describing the state as given above
    #   network_ref: a reference to the network containing this neuron
    def initialize(state_vec, network_ref)
      # Set state vector
      @state_vector = state_vec

      # Reserve space for other instance variables
      @out_conns = Hash.new
      @input_sum_terms = []
      @network = network_ref
    end

    #### Accessor methods for individual elements of the state vector ####

    def set_a(new_a)
      @state_vector[0] = new_a
    end

    def set_b(new_b)
      @state_vector[1] = new_b
    end

    def set_x(new_x)
      @state_vector[2] = new_x
    end

    def set_x_prime(new_x_prime)
      @state_vector[3] = new_x_prime
    end

    def set_x_dbl_prime(new_x_dbl_prime)
      @state_vector[4] = new_x_dbl_prime
    end

    def get_a
      return @state_vector[0] 
    end

    def get_b
      return @state_vector[1]
    end

    def get_x
      return @state_vector[2]
    end

    def get_x_prime
      return @state_vector[3]
    end

    def get_x_dbl_prime
      return @state_vector[4] 
    end

    def get_layer
      return @state_vector[5]
    end

    # Updates the current state of an input node using exact solutions to the equation
    # x_dbl_prime = -a*x (since there are no inputs to the input nodes, the added term is zero)
    # The exact solution is x = A*sin(sqrt(a)*t + phi), where A = sqrt(x_prime^2+x^2), phi = arctan(x/x_prime)
    def update_input_state
      amp = GSL::hypot(get_x_prime, get_x)
      phi = arctan(get_x/get_x_prime)
      new_x = amp*sin(sqrt(get_a)*@network.get_current_time + phi)
      set_x(new_x)
    end

    # Updates the current state based on the current states of inputs to this node.  
    def update_state

      if(get_layer == 0)
        update_input_state
        return
      end

      # Calculate sum of inputs
      sum = @input_sum_terms.inject(0){|sum,item| sum+item}

      # An oscillator
      #   a: spring constant
      #   b: damping
      #   sum: sum of external forces, weighted by connection strength

      dim = 2

      # Setup system of ODEs to solve
      # x[0]: displacement, x[1]: velocity
      func = Proc.new { |t, x, dxdt, params|
        b = params[1] 
        sum = params[2] 
        a = params[3]
        dxdt[0] = x[1]
        dxdt[1] = (sum - b*x[1] - a*x[0])
      }

      # Create solver
      gos = GSL::Odeiv::Solver.alloc(GSL::Odeiv::Step::RKF45, [1e-6, 0.0], func, dim)

      # Set parameters for solving
      gos.set_params(get_b, sum, get_a)
      t = @network.get_current_time 
      t1 = t + @network.get_time_step
      tend = 10.0
      h = 1e-6

      # Initial guess vector
      x = GSL::Vector.alloc([get_x_prime, get_x_dbl_prime])

      GSL::ieee_env_setup()

      # Apply solver
      t, h, status = gos.apply(t, t1, h, x)
      set_x(x[0])
      set_x_prime(x[1])
    end

    # Propagates the node's current x to all of its out_conns
    def propagate
      # Iterate through all outgoing connections
      @out_conns.each_key do |receiver|
        # Calculate the term of the sum corresponding to the propagating node
        term = @out_conns[receiver] * (self.get_x - receiver.get_x)
        # Insert the term in the receiver's registry
        receiver.input_sum_terms << term
      end
    end

  end

end
