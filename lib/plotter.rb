# Interfaces to the GNU graph utility to create plots from GSL::Vectors

require 'gsl'

module Plot
  
 class Plotter
     # Creates a plot-suitable data file of ordered pairs
     #   filename: a String giving the filename
     #   data1, data2: GSL::Vectors of data, equal length preferably
     def make_plottable_data(filename, data1, data2)
       file = File.open("#{filename}", 'w+')
       data1.len.times do |index|
         file.puts(data1[index].to_s + " " + data2[index].to_s + "\n")
       end
       file.close
     end

     # Plots data using GNU graph
     #   options: a GNU graph options string, including output filetype
     #   input_filename: name of an input data file
     #   output_filename: name of file to store the graph in (may be left off if not saving graph)
     def make_plot(options, input_filename, output_filename=nil)
       if output_filename
         `graph #{options} < #{input_filename} > #{output_filename}`
       else
         `graph #{options} < #{input_filename}`
       end
     end
 end

end
