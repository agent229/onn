require File.expand_path(File.dirname(__FILE__)) + "/../lib/ga_onn"
include GAONN

  @node_data = GSL::Matrix[[0.7,0,1,0,0,0],[0.3,0.1,0,0,0,1],[0.8,0.1,0,0,0,1],[0.2,0.1,0,0,0,2],[0.7,0.1,0,0,0,2]]
  @conns = GSL::Matrix[[0,0.4,0.4,0,0],[0,0,0.2,0.4,-0.6],[0,0.2,0,-0.6,0.4],[0,0,0,0,0],[0,0,0,0,0]]
  @inputs = []
  srand(1)
  10.times do |index|
    input_set = GSL::Matrix.calloc(1,5)
    input_set[0,0] = rand*3
    input_set[0,2] = rand*2
    @inputs << input_set
  end
  @ga = GA.new(@node_data,@conns,@inputs,2,0,100,200,0.75,1.5)
  best_chrom, fitness = @ga.run
  puts "best chromosome data: "
  puts best_chrom.node_data

  @net = ONN.new(@inputs,best_chrom.node_data,@conns,2,1,25,0.2)

  amps1 = []
  amps2 = []

  @net.eval_over_time
  @net.nodes.size.times do |index|
    @net.plot_x_over_time(index)
  end

  amp1, freq1 = @net.fourier_analyze(3)
  amps1 << amp1     
  amp2, freq2 = @net.fourier_analyze(4)
  amps2 << amp2

  @inputs.size.times do |index|
    @net.set_input(index)
    @net.eval_over_time
    amp1, freq1 = @net.fourier_analyze(3)
    amp2, freq2 = @net.fourier_analyze(4)
    amps1 << amp1
    amps2 << amp2
  end

  input_a_vals = []
  @inputs.each do |input_set|
    input_set.each_row do |input|
      input_a_vals << input[0]
    end
  end
  GSL::graph(input_a_vals.to_gv,amps1.to_gv,amps2.to_gv,"-S 16 -m -2 -T png -C -X 'input a' -Y 'output amps' -L 'best chromosome output amps vs input a' > ga_output_amps.png")
