# This is an implementation of oscillator neural networks.
# These networks are made of nodes which are harmonic oscillators.
# The nodes are capable of updating their own states based on
# inherent frequency, damping, and inputs from other nodes. 
# The ONN class describes an entire network object, and
# the OscillatorNode class describes a single node.
# By importing the OscillatorNeuralNetwork module, you can create and
# run an ONN and plot data and observe behavior.

module OscillatorNeuralNetwork

  require 'gsl' # For Ruby/GSL scientific library (vectors, matrices, graphing)

  class ONN 
  
    attr_accessor :nodes          # An array of OscillatorNode objects
    attr_reader :t_step           # time step
    attr_reader :eval_steps       # number of steps to run net
    attr_reader :connections      # Connections Matrix
    attr_reader :curr_time        # Current simulated time
    attr_reader :curr_step        # Current time step number

    DEFAULT_NUM_EVALS_PARAM = 500 # Parameter helping decide how many evaluations to get the network to a stable state

#### Class methods ####

    # Mutation function. Randomly perturbs the network's configuration states with a given chance.
    #  chromosome:    a Network to mutate
    #  mutation_rate: parameter describing the rate of mutation (chance of random mutation)
    # Returns the mutated/perturbed Network. 
    def self.mutate(network, mutation_rate)
      # TODO write mutation--take hash of options?
    end

#### Initialization ####

    # Initializes an network of coupled harmonic oscillators. 
    #   input_list:      a list of input data representing different sets of inputs to train the network on. 
    #                    Should be a GSL::Matrix with rows representing the input nodes, cols their different values
    #   node_data:       a GSL::Matrix containing row vectors of initial node data (see OscillatorNode class for detail) 
    #   connections:     a GSL::Matrix of connection strengths. ijth entry is connection from node i to node j
    #   num_outputs:     the number of outputs in the network
    #   num_inputs:      the number of inputs in the network
    #   num_evals_param: (optional) parameter used to decide how many evaluations to complete before evaluating outputs
    def initialize(input_list, node_data, connections, num_outputs, num_inputs, num_evals_param=DEFAULT_NUM_EVALS_PARAM)
      @input_list = input_list.clone
      @node_data = node_data.clone
      @num_outputs = num_outputs 
      @num_inputs = num_inputs
      @num_evals_param = num_evals_param
      @connections = connections.clone                
      @nodes = create_node_list(@num_inputs, node_data) 
      set_input(0)                         
    end

    # Creates a list of OscillatorNode objects which contain the data. 
    #   num_inputs: the number of input nodes to reserve space for
    #   node_data:  GSL::Matrix with rows containing node data vectors (see OscillatorNode for detail)
    # Returns an Array of initialized OscillatorNodes. 
    def create_node_list(num_inputs, node_data)
      nodes = []
      empty_vector = GSL::Vector.calloc(node_data.size2)
      num_inputs.times do
        nodes << OscillatorNode.new(empty_vector,self)
      end
      node_data.each_row do |node_datum|
        nodes << OscillatorNode.new(node_datum, self) # Initialize node states
      end
      nodes = set_conns_from_mat(nodes)               # Set connections
      return nodes
    end

    # Sets a new input set into the network and sets it up to run
    def set_input(index)
      new_input_vals = []
      @input_list.each do |input|
        new_input_vals << input.row(index) 
      end
      new_input_vals.each_index do |input|
        @nodes[input].states_matrix.row(0) = input
      end
      @t_step, @eval_steps = calc_time_vars 
      @curr_time = 0.0                                              
      @curr_step = 0
    end

#### Evaluation ####

    # Evaluates the network over time over one input (the one currently encoded in the network). 
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

#### Custom accessor methods ####

    # Calculates a suitable time step to use based on the Nyquist-Shannon Sampling Theorem. 
    # Also calculates the number of time steps over which to evaluate the network.
    # Returns both values in this order: t_step, eval_steps
    def calc_time_vars
      a_vals_arr = []
      @nodes.each do |node|
        a_vals_arr << node.states_matrix[0,0]
      end
      a_vals = a_vals_arr.to_gv
      freq_vals = a_vals.sqrt/2*GSL::M_PI
      ones = GSL::Vector.alloc(freq_vals.len).set_all(1)
      quotients = ones/(2*freq_vals)
      min_quotient = quotients.min
      t_step = 0.4*min_quotient
      periods = quotients*2 
      max_period = periods.max
      return t_step, ((max_period*@eval_steps_param).round-1)
    end

    # Stores connection information from the connections matrix into the nodes.
    #   nodes: list of OscillatorNode objects in order
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

    def get_time_step
      return @t_step
    end

    def get_current_time
      return @curr_time
    end

    def get_curr_step
      @curr_step
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
   
    # Plots a node's fourier transform after a full run of the network.
    def plot_fourier(node_index)
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
      GSL::graph(f,fft_norm.abs, "-T png -C -X 'Frequency (Hz)' -Y 'Amplitude' -L 'Node #{node_index} Scaled FFT' > fft#{node_index}.png")
    end

#### Misc. helper functions ####
    
    def increment_time
      @curr_time += @t_step
      @curr_step += 1
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

  end

  # This class describes a single OscillatorNode. Each node knows everything about
  # its own natural state, current state, next state, and inbound/outbound connections.
  # Variable state_vector is a GSL::vector containing the following information:
  # <a, b, x, x_prime, x_dbl_prime, layer> where the equation describing the
  # oscillator is x_dbl_prime = -a*x - b*x_prime + input_sum
  # so that a = spring constant, b = damping coefficient

  class OscillatorNode

    attr_accessor :states_matrix   # Matrix containing state row vectors over time (first row = first state)
    attr_accessor :input_sum_terms # Sum of current inputs to node
    attr_accessor :out_conns       # Hash of outgoing connections
    attr_accessor :layer           # The layer number of this node

    # Globally available system of ODEs describing oscillators
    # x[0]: displacement, x[1]: velocity
    $func = Proc.new { |t, x, dxdt, params|
      b = params[0] 
      sum = params[1] 
      a = params[2]
      dxdt[0] = x[1]
      dxdt[1] = (sum - b*x[1] - a*x[0])
    }

    # Initialize a new OscillatorNode by passing a state vector and reference to the network.
    #   state_vec:   a GSL::Vector describing the state as given above
    #   network_ref: a reference to the network containing this neuron
    def initialize(state_vec, network_ref)
      @states_matrix = GSL::Matrix.alloc(network_ref.eval_steps, state_vec.len-1)
      @layer = state_vec.pop.to_i
      @states_matrix.set_row(0, state_vec)
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

    # Updates the current state of an input node using exact solutions to the equation
    #   x_dbl_prime = -a*x 
    # The exact solutions are:
    #   x           = A*sin(sqrt(a)*t+phi), 
    #   x_prime     = A*sqrt(a)*cos(sqrt(a)*t+phi) 
    #   x_dbl_prime = -A*a*sin(sqrt(a)*t+phi) 
    #   where A = sqrt(x_prime^2+x^2), phi = arctan(x/x_prime)
    # Stores all of the new states in the next state vector
    # TODO FIXME 
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

    # Propagates the node's current x to all of its out_conns
    def propagate
      curr_step = @network.get_curr_step
      @out_conns.each_key do |receiver|
        term = @out_conns[receiver] * (get_x(curr_step)-receiver.get_x(curr_step))
        receiver.input_sum_terms << term
      end
    end

  end

end
