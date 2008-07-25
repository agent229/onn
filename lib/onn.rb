# This is an implementation of oscillator neural networks which can update its own state, 
# plot various things about its state.

module OscillatorNeuralNetwork
 
  # For genetic algorithm
  require File.expand_path(File.dirname(__FILE__)) + "/onn_genetic_algorithm"
  include ONNGeneticAlgorithm      

  # For Ruby/GSL scientific library (vectors, matrices, graphing)
  require 'gsl'

  class ONN 
  
    attr_accessor :nodes        # An array of OscillatorNode objects
    attr_reader :t_step         # time step
    attr_reader :eval_steps     # number of steps to run net
    attr_reader :seed           # PRNG seed governing all uses of "rand"
    attr_reader :connections    # Connections Matrix
    attr_reader :curr_time      # Current simulated time
    attr_reader :curr_step      # Current time step

    DEFAULT_MUTATION_RATE = 0.4
    DEFAULT_NUM_EVALS_PARAM = 500 
    DEFAULT_SEED = 0

    # Initializes an ONN of coupled harmonic oscillators. 
    #   node_data:       a GSL::Matrix containing vectors of node data (see OscillatorNode class for detail) 
    #   connections:     a GSL::Matrix of connection strengths. ijth entry is connection from node i to node j
    #   num_outputs:     the number of outputs that will exist in the network
    #   seed:            a PRNG seed governing all PRNG uses in this run (for repeatability)
    #   num_evals_param: parameter used to decide how many evaluations to complete before evaluating outputs
    def initialize(node_data, connections, num_outputs, seed=DEFAULT_SEED, num_evals_param=DEFAULT_NUM_EVALS_PARAM)
      @seed = seed
      @num_outputs = num_outputs
      @connections = connections.clone                                              # Store connections GSL::Matrix
      @t_step, @eval_steps = calc_time_vars(node_data,num_evals_param) # Calculate appropriate time step, number of time steps
      @nodes = create_node_list(node_data)                                          # Initialize network of nodes by layer
      @curr_time = 0.0                                                              # Set current time to 0
      @curr_step = 0
      srand(seed)                                                                   # Seed PRNG for this run
    end

    # Creates the list of OscillatorNode objects which contain the data 
    #   node_data:   GSL::Matrix with rows containing node data vectors (see OscillatorNode for detail)
    # Returns the list of nodes.
    def create_node_list(node_data)
      nodes = []
      node_data.each_row do |node_datum|
        nodes << OscillatorNode.new(node_datum, self) # Initialize node states
      end
      nodes = set_conns_from_mat(nodes)   # Set connections
      return nodes
    end

    # Evaluates the network over time by calling "eval" repeatedly. 
    # The number of times it is called is determined by parameters and
    # initial conditions of the system (stored in @eval_steps)
    def eval_over_time
      (@eval_steps-1).times do eval end 
    end

    # Evaluates the network's output state by propagating the current input states through the network.
    # Evaluates over one time step, then increments the time step after updating the states.
    def eval
      @nodes.each do |node|  # Each node tells its state to nodes it is connected to
        node.propagate
      end
      @nodes.each do |node|
        node.update_state    # Calculate and update the node states
      end
      increment_time         # Increment the time
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

#### Plotting functions ####

    # Plots a node's x values over time
    #   data: the GSL::Matrix describing the node's history
    def plot_x_over_time(node_num)
      data = @nodes[node_num].states_matrix
      x_vals = data.col(2) 
      t = GSL::Vector.linspace(0,@eval_steps*@t_step,@eval_steps)
      x_vals.graph(t,"-T png -C -X 'Time' -Y 'X' -L 'Waveform: Node #{node_num}' > xvals#{node_num}.png")
    end
    
    def plot_fourier(freq_vec,fft,node_index)
      GSL::graph(freq,fft.abs, "-T png -C -X 'Frequency (Hz)' -Y 'Amplitude' -L 'Node #{node_index} Scaled FFT' > fft#{node_index}.png")
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

    # Calculates good guesses of a time step to use based on the minimum a (spring constant)
    # and the number of steps to evaluate the network until returning the output states.
    # Returns both values in the order t_step, eval_steps
    def calc_time_vars(node_data,eval_steps_param)
      a_vals = node_data.col(0)
      freq_vals = a_vals.sqrt/2*GSL::M_PI
      ones = GSL::Vector.alloc(freq_vals.len).set_all(1)
      quotients = ones/(2*freq_vals)
      min_quotient = quotients.min
      t_step = 0.4*min_quotient
      periods = quotients*2 
      max_period = periods.max
      return t_step, ((max_period*eval_steps_param).round-1)
    end

    # Stores connection information from the connections matrix into the nodes.
    #   nodes:       list of OscillatorNode objects in order
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

      # Select roughly the last 2/3 of the data to analyze to avoid transients
      offset = (states.size1/3).floor
      subvec_len = 2*offset
      x_vals = states.col(2).transpose.subvector(offset,subvec_len)

      n = x_vals.len
      fs = 1/@t_step
      k = GSL::Vector.indgen(n)
      t = n/fs
      freq = k/t
      fft = x_vals.fft
      fft2 = fft.subvector(1,n-2).to_complex2
      fft_norm = fft2/fft2.len

      f = GSL::Vector.linspace(0,fs/2,fft_norm.size) 
      dom_amp = fft_norm.abs.max
      dom_freq = f[fft_norm.abs.max_index]

      return dom_amp, dom_freq
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
      return Math::sqrt(node_data_vec[0])/2*GSL::M_PI
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

  # This class describes a single OscillatorNode. Each neuron knows everything about
  # its own natural state, current state, next state, and inbound/outbound connections.
  # Variable state_vector is a GSL::vector containing the following information:
  # <a, b, x, x_prime, x_dbl_prime, layer> where the equation describing the
  # oscillator is x_dbl_prime = -a*x - b*x_prime + input_sum
  # so that a = spring constant, b = damping coefficient

  class OscillatorNode

    attr_accessor :states_matrix
    attr_accessor :input_sum_terms
    attr_accessor :out_conns
    attr_accessor :layer

    # Setup system of ODEs to solve
    # x[0]: displacement, x[1]: velocity
    $func = Proc.new { |t, x, dxdt, params|
      b = params[0] 
      sum = params[1] 
      a = params[2]
      dxdt[0] = x[1]
      dxdt[1] = (sum - b*x[1] - a*x[0])
    }

    # Initialize a new OscillatorNode by passing a "natural state" hash.
    #   state_vec: a GSL::Vector describing the state as given above
    #   network_ref: a reference to the network containing this neuron
    def initialize(state_vec, network_ref)
      # Set state vector as first row in states_matrix
      @states_matrix = GSL::Matrix.alloc(network_ref.eval_steps, state_vec.len-1)
      @layer = state_vec.pop.to_i
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

#### State update rules ####

    # Updates the current state of an input node using exact solutions to the equation
    #   x_dbl_prime = -a*x 
    # The exact solutions are:
    #   x           = A*sin(sqrt(a)*t+phi), 
    #   x_prime     = A*sqrt(a)*cos(sqrt(a)*t+phi) 
    #   x_dbl_prime = -A*a*sin(sqrt(a)*t+phi) 
    #   where A = sqrt(x_prime^2+x^2), phi = arctan(x/x_prime)
    # Stores all of the new states in the next state vector
    # TODO Why isn't this working???
    def update_input_state
      last_time_step = @network.get_curr_step
      next_time_step = last_time_step + 1
      
      a = get_a(last_time_step)
      b = get_b(last_time_step)
      x = get_x(last_time_step)
      x_prime = get_x_prime(last_time_step)
      t = @network.get_current_time
      t_next = t + @network.get_time_step

      amp = GSL::hypot(x_prime,x)
      phi = Math::atan(x/x_prime)

      new_x = amp*Math::sin(Math::sqrt(a)*t_next+phi)
      new_x_prime = amp*Math::sqrt(a)*Math::cos(Math::sqrt(a)*t_next+phi)
      # new_x_dbl_prime = -a*new_x

      set_x(next_time_step,new_x)
      set_x_prime(next_time_step,new_x_prime)
      # set_x_dbl_prime(next_time_step,new_x_dbl_prime)
      set_a(next_time_step,a)
      set_b(next_time_step,b)
    end

    # Updates the current state based on the current states of inputs to this node.  
    def update_state

      # TODO uncomment once update_input_state works
      # if(@layer == 0)
      #   update_input_state
      #   return
      # end

      # Store time step indices
      last_time_step = @network.get_curr_step
      next_time_step = last_time_step + 1 

      # Calculate sum of inputs
      sum = @input_sum_terms.inject(0){|sum,item| sum+item}

      # An oscillator
      #   a: spring constant
      #   b: damping constant
      #   sum: sum of external forces (weighted by connection strength)

      # Dimension of the ODE system
      dim = 2

      # Create solver
      eps_params = [1e-6, 0.0]
      gos = GSL::Odeiv::Solver.alloc(GSL::Odeiv::Step::RKF45, eps_params, $func, dim)

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
      # set_x_dbl_prime(next_time_step,sum-b*x[1]-a*x[0])
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
