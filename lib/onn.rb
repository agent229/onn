# This is an implementation of oscillator neural networks which uses a genetic
# algorithm as the learning rule. It learns by adjusting either individual connection weights
# or the "natural" properties of the oscillators, depending on the need of the user.
# (Loosely) based on the neural network from the ai4r library.

module OscillatorNeuralNetwork
 
  # For genetic algorithm
  require File.expand_path(File.dirname(__FILE__)) + "/onn_genetic_algorithm"
  include ONNGeneticAlgorithm      

  # For Ruby/GSL scientific library (vectors, matrices, graphing)
  require 'gsl'

  # "Genetic algorithm oscillator neural network" (GAONN) class definition describes an
  # entire network object through a list of nodes and some parameters
  class GAONN 
  
    attr_accessor :nodes        # An array of OscillatorNeuron objects
    attr_reader :t_step         # time step
    attr_reader :eval_steps     # number of steps to run net
    attr_reader :seed           # PRNG seed governing all uses of "rand"
    attr_reader :connections    # Connections Matrix
    attr_reader :curr_time      # Current simulated time
    attr_reader :curr_step      # Current time step

    DEFAULT_MUTATION_RATE = 0.4
    DEFAULT_T_STEP_PARAM = 0.02    # Default/suggested settings for parameters
    DEFAULT_NUM_EVALS_PARAM = 300
    DEFAULT_SEED = 0

#### Class method(s) ####

    # This method trains a network using an instance of an OscillatorGeneticAlgorithm.
    #   network:       the initial network to train 
    #   input:         GSL::Matrix of inputs to the network 
    #   exp_output:    GSL::Matrix of expected/desired output for the given input 
    #   pop_size:      How many solutions to keep in the "potential solution" population
    #   gens:          How many generations to run the GA
    #   mutation_rate: a parameter describing how frequently a given solution gets random mutations
    # Returns an array containing the evolved network  along with a weighted error estimate of 
    # how "close" the expected and actual outputs are after training is complete.
    def self.train(network, input, exp_output, pop_size, gens, mutation_rate=DEFAULT_MUTATION_RATE)
      change_input(input)                                            # Set new input states
      ga = GeneticSearch.new(network, pop_size, gens, mutation_rate) # Create GA
      best_net = ga.run                                              # Run GA
      # TODO determine exactly what gets returned from GA, what is neccessary to run it for error
      # TODO HERE, do the loop for evaluation... perhaps abstract somewhere to use other places too
      # (eval only evaluates once, decide here how many times to do this, for how long, fourier etc)
      err = weighted_error(exp_output)
      return [best_net, err]
    end

#### Instance methods ####

    # Initializes an ONN of coupled harmonic oscillators. 
    #   node_data:       a GSL::Matrix containing vectors of node data (see OscillatorNeuron class for detail) 
    #   connections:     a GSL::Matrix of connection strengths. ijth entry is connection from node i to node j
    #   num_outputs:     the number of outputs that will exist in the network
    #   seed:            a PRNG seed governing all PRNG uses in this run (for repeatability)
    #   t_step_param:    parameter to be used in simulation to help decide time step (scales the minimum period)
    #   num_evals_param: parameter used to decide how many evaluations to complete before evaluating outputs
    def initialize(node_data, connections, num_outputs, seed=DEFAULT_SEED, t_step_param=DEFAULT_T_STEP_PARAM, num_evals_param=DEFAULT_NUM_EVALS_PARAM)
      @seed = seed
      @num_outputs = num_outputs
      @connections = connections.clone                                              # Store connections GSL::Matrix
      @t_step, @eval_steps = calc_time_vars(node_data,t_step_param,num_evals_param) # Calculate appropriate time step, number of time steps
      @nodes = create_node_list(node_data)                                          # Initialize network of nodes by layer
      @curr_time = 0.0                                                              # Set current time to 0
      @curr_step = 0
      srand(seed)                                                                   # Seed PRNG for this run
    end

    # Creates the list of OscillatorNeuron objects which contain the data 
    #   node_data:   GSL::Matrix with rows containing node data vectors (see OscillatorNeuron for detail)
    # Returns the list of nodes.
    def create_node_list(node_data)
      nodes = []
      node_data.each_row do |node_datum|
        nodes << OscillatorNeuron.new(node_datum, self) # Initialize node states
      end
      nodes = set_conns_from_mat(nodes)   # Set connections
      return nodes
    end

    # Evaluates the network over time by calling "eval" repeatedly. 
    # The number of times it is called is determined by parameters and
    # initial conditions of the system (stored in @eval_steps)
    def eval_over_time
      (@eval_steps-1).times { eval } 
    end

    # Evaluates the network's output state by propagating the current input states through the network.
    # Evaluates over one time step, then increments the time step after updating the states.
    def eval
      @nodes.each do |node|         # Each node tells its state to nodes it is connected to
        node.propagate
      end
      @nodes.each do |node|
        node.update_state           # Calculate and update the node states
      end
      @curr_time = increment_time   # Increment the time
    end

    # Error weighting function. Calculates a Euclidean-style weighted error measure of the output.
    #   expected: the expected/desired result (GSL::Matrix of data)
    def weighted_error(expected)
      w_err = 0.0
      result_amps = result_freqs = []
      @nodes.size-@num_outputs..@nodes.size do |index|
        result_amps_i, result_freqs_i = fourier_analyze(index)
        result_amps << result_amps_i
        result_freqs << result_freqs_i
      end
      result_amps.each_index do |node_index|
        amp_term = result_amps[node_index] - amp_from_vec(expected[node_index])
        freq_term = result_freqs[node_index] - freq_from_vec(expected[node_index])
        w_err += GSL::hypot(amp_term, freq_term)
      end
      w_err /= result_amps.length
      return w_err
    end

#### GA-related functions ####

    # GA fitness function. Takes a nodelist and returns a weighted error calculation of the result
    # compared with the expected/desired result by evaluating the network.
    #   chromosome: a list of OscillatorNeurons 
    # TODO fix
    def fitness(chromosome)
      @nodes = chromosome
      output = eval_over_time
      err = weighted_error(output,@curr_expected)
      return err
    end

    # Mutation function. Randomly mutates with a given chance specified by GA.
    #  chromosome: a nodelist (list of Oscillator Neurons divided into layers)
    #  mutation_rate: parameter describing the rate of mutation (chance of random mutation)
    # Returns the mutated chromosome
    # TODO fix
    def mutate(chromosome, mutation_rate)
      chromosome.each do |node|
          # Add random mutations with chance mutation_rate
          if rand < mutation_rate
            # TODO fix scale, make more general (only changing a for now)
            node.set_a(node.get_a + rand - 0.5)
          end
      end
      return chromosome
    end

#### Miscellaneous helper functions ####

    # Returns an array of states matrices from nodes beginning at beg_ind and ending at
    # end_ind. If the arguments are left off, returns all state matrices. Preserves order.
    def get_states(beg_ind=0, end_ind=@nodes.length)
      states = []
      (end_ind-beg_ind).times do |ind|
        states << @nodes[ind].states_matrix 
      end
      return states
    end
 
    # Plots a node's x values over time
    #   data: the GSL::Matrix describing the node's history
    def plot_x_over_time(node_num)
      data = @nodes[node_num].states_matrix
      x_vals = data.col(2) 
      t = GSL::Vector.linspace(0,@eval_steps*@t_step,@eval_steps)
      x_vals.graph(t,"-T png -C -L 'Waveform: Node #{node_num}' > xvals#{node_num}.png")
    end

    # Calculates good guesses of a time step to use based on the minimum a (spring constant)
    # and the number of steps to evaluate the network until returning the output states.
    # Returns both values in the order t_step, eval_steps
    def calc_time_vars(node_data,t_step_param,eval_steps_param)
      a_vals = node_data.col(0)
      a_inv_vals = GSL::Vector.alloc(a_vals.len)
      ind = 0
      a_vals.each do |val|
        a_inv_vals[ind] = 1/val
        ind += 1
      end
      periods = 2*GSL::M_PI*a_inv_vals.sqrt
      min_period = periods.min 
      max_period = periods.max
      return (min_period * t_step_param), ((max_period*eval_steps_param).round-1)
    end

    # Stores connection information from the connections matrix into the nodes.
    #   nodes:       list of OscillatorNeuron objects in order
    # Returns the nodelist with connections set
    def set_conns_from_mat(nodes)
      pointer_index = 0
      receiver_index = 0
      @connections.each_row do |pointer|
        pointer.each do |receiver|
          if !GSL::equal?(receiver, 0.0)
            nodes[pointer_index].out_conns[nodes[receiver_index]] = receiver
          end
          receiver_index += 1
        end
        receiver_index = 0
        pointer_index += 1
      end
      return nodes
    end

    # Sets the input nodes to different oscillator data
    #   new_input_data: a GSL::Matrix of oscillator data vectors, one for each input node, in order
    def change_input(new_input_data)
      node_counter = 0
      new_input_data.each_row do |row|
        @nodes[node_counter].state_matrix.set_row(get_curr_step, row.clone)
        node_counter += 1
      end
    end

    # Uses fourier/wavelet transform to get dominant frequency, amplitude
    #   node_index: the index of the node to fourier analyze over time
    def fourier_analyze(node_index)
      states = @nodes[node_index].states_matrix
      x_vals = states.col(2).transpose

      fft = x_vals.fft
      fft2 =  fft.subvector(1,x_vals.len-1).to_complex2
      amp = fft2.abs
      freq = fft2.arg
 
      # Graph for inspection
      f = GSL::Vector.linspace(0, 10, 10)
      fgraph(amp, "-T png -C -L 'Node #{node_index}: Frequency [Hz]' > fft#{node_index}.png")

      return amp, freq
    end

    # Calculates a wave's amplitude based on its state vector
    #   node_data_vec: a node data vector
    # Returns the wave's amplitude
    def amp_from_vec(node_data_vec)
      return GSL::hypot(node_data_vec[3],node_data_vec[2]) 
    end

    # Calculates a wave's frequency based on its state vector
    #   node_data_vec: a node data vector
    # Returns the wave's frequency 
    def freq_from_vec(node_data_vec)
      return Math::sqrt(node_data_vec[0]) 
    end

    def get_time_step
      return @t_step
    end

    def get_current_time
      return @curr_time
    end

    def increment_time
      @curr_time += @t_step
      @curr_step += 1
    end

    def get_curr_step
      @curr_step
    end 

  end

  # This class describes a single OscillatorNeuron. Each neuron knows everything about
  # its own natural state, current state, next state, and inbound/outbound connections.
  # Variable state_vector is a GSL::vector containing the following information:
  # <a, b, x, x_prime, x_dbl_prime, layer> where the equation describing the
  # oscillator is x_dbl_prime = -a*x - b*x_prime + input_sum
  # so that a = spring constant, b = damping coefficient

  class OscillatorNeuron

    attr_accessor :states_matrix
    attr_accessor :input_sum_terms
    attr_accessor :out_conns
    attr_accessor :layer

    # Initialize a new OscillatorNeuron by passing a "natural state" hash.
    #   state_vec: a GSL::Vector describing the state as given above
    #   network_ref: a reference to the network containing this neuron
    def initialize(state_vec, network_ref)
      # Set state vector as first row in states_matrix
      @states_matrix = GSL::Matrix.alloc(network_ref.eval_steps, state_vec.len-1)
      @layer = state_vec.pop 
      @states_matrix.set_row(0, state_vec)

      # Reserve space for other instance variables
      @out_conns = Hash.new
      @input_sum_terms = []
      @network = network_ref
    end

    #### Accessor methods for individual elements of the state matrix ####

    def set_a(step_num, new_a)
      @states_matrix[step_num][0] = new_a
    end

    def set_b(step_num, new_b)
      @states_matrix[step_num][1] = new_b
    end

    def set_x(step_num, new_x)
      @states_matrix[step_num][2] = new_x
    end

    def set_x_prime(step_num, new_x_prime)
      @states_matrix[step_num][3] = new_x_prime
    end

    def set_x_dbl_prime(step_num, new_x_dbl_prime)
      @states_matrix[step_num][4] = new_x_dbl_prime
    end

    def get_a(step_num)
      return @states_matrix[step_num][0] 
    end

    def get_b(step_num)
      return @states_matrix[step_num][1]
    end

    def get_x(step_num)
      return @states_matrix[step_num][2]
    end

    def get_x_prime(step_num)
      return @states_matrix[step_num][3]
    end

    def get_x_dbl_prime(step_num)
      return @states_matrix[step_num][4] 
    end

    # Updates the current state of an input node using exact solutions to the equation
    #   x_dbl_prime = -a*x 
    # The exact solutions are:
    #   x           = A*sin(sqrt(a)*t+phi), 
    #   x_prime     = A*sqrt(a)*cos(sqrt(a)*t+phi) 
    #   x_dbl_prime = -A*a*sin(sqrt(a)*t+phi) 
    #   where A = sqrt(x_prime^2+x^2), phi = arctan(x/x_prime)
    # Stores all of the new states in the next state vector
    def update_input_state
      last_time_step = @network.get_curr_step
      next_time_step = last_time_step + 1
      
      a = get_a(last_time_step)
      b = get_b(last_time_step)
      x = get_x(last_time_step)
      x_prime = get_x_prime(last_time_step)
      t = @network.get_current_time

      amp = GSL::hypot(x_prime, x)
      phi = Math::atan(x/x_prime)

      new_x = amp*Math::sin(Math::sqrt(a)*t+phi)
      new_x_prime = amp*Math::sqrt(a)*Math::cos(Math::sqrt(a)*t+phi)
      new_x_dbl_prime = -amp*a*Math::sin(Math::sqrt(a)*t+phi)

      set_x(next_time_step,new_x)
      set_x_prime(next_time_step,new_x_prime)
      set_x_dbl_prime(next_time_step,new_x_dbl_prime)
      set_a(next_time_step,a)
      set_b(next_time_step,b)
    end

    # Updates the current state based on the current states of inputs to this node.  
    def update_state

      if(@layer == 0)
        update_input_state
        return
      end

      # Store time step indices
      last_time_step = @network.get_curr_step
      next_time_step = last_time_step + 1 

      # Calculate sum of inputs
      sum = @input_sum_terms.inject(0){|sum,item| sum+item}

      # An oscillator
      #   a: spring constant
      #   b: damping constant
      #   sum: sum of external forces (weighted by connection strength)

      # Setup system of ODEs to solve
      # x[0]: displacement, x[1]: velocity
      func = Proc.new { |t, x, dxdt, params|
        b = params[0] 
        sum = params[1] 
        a = params[2]
        dxdt[0] = x[1]
        dxdt[1] = (sum - b*x[1] - a*x[0])
      }
     
      # Dimension of the ODE system
      dim = 2

      # Create solver
      eps_params = [1e-6, 0.0]
      gos = GSL::Odeiv::Solver.alloc(GSL::Odeiv::Step::RKF45, eps_params, func, dim)

      # Set parameters for solving
      a = get_a(last_time_step)
      b = get_b(last_time_step)
      gos.set_params(b, sum, a)
      t = @network.get_current_time 
      t1 = t + @network.get_time_step
      h = 1e-6

      # Initial conditions vector (values from the last time step)
      x = GSL::Vector[get_x(last_time_step),get_x_prime(last_time_step)]

      GSL::ieee_env_setup()

      # Apply solver
      while t < t1
        t, h, status = gos.apply(t, t1, h, x)
        break if status != GSL::SUCCESS
      end

      # Set new state variables
      set_a(next_time_step,a)
      set_b(next_time_step,b)
      set_x(next_time_step,x[0])
      set_x_prime(next_time_step,x[1])
      set_x_dbl_prime(next_time_step,sum-b*x[1]-a*x[0])
      @input_sum_terms = []
    end

    # Propagates the node's current x to all of its out_conns
    def propagate
      curr_step = @network.get_curr_step
      # Iterate through all outgoing connections
      @out_conns.each_key do |receiver|
        # Calculate the term of the sum corresponding to the propagating node
        term = @out_conns[receiver] * (get_x(curr_step)-receiver.get_x(curr_step))
        # Insert the term in the receiver's registry
        receiver.input_sum_terms << term
      end
    end

  end

end
