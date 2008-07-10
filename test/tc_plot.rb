# Tests the plotting methods
require '~/Documents/SFI/NN/onn/lib/plotter'
include Plot
require 'gsl'
require 'test/unit'

class TestPlot < Test::Unit::TestCase

  def setup
    @v1 = GSL::Vector.linspace(0, 2.0*GSL::M_PI, 20)
    @v2 = GSL::Sf::sin(@v1)
    @v3 = 2 * GSL::Sf::sin(@v1)
    @plotter = Plotter.new
  end

  def test_simple_plotting
    @plotter.make_plottable_data("data_test",@v1,@v2)
    @plotter.make_plot("-T png","data_test","plot_test.png")
    @plotter.make_plottable_data("data_test2",@v1,@v3)
    @plotter.make_plot("-T png","data_test2","plot_test2.png")
  end

  def test_plotting_fft
    n = 2048
    sampling = 1000   # 1 kHz
    tmax = 1.0/sampling*n
    freq1 = 50
    freq2 = 120
    t = GSL::Vector.linspace(0, tmax, n)

    x1 = GSL::Sf::sin(2*GSL::M_PI*freq1*t)
    y1 = x1.fft

    x2 = 2 * GSL::Sf::sin(2*GSL::M_PI*freq1*t)
    y2 = x2.fft

    x3 = 0.667 * GSL::Sf::sin(2*GSL::M_PI*freq1*t)
    y3 = x3.fft

    x4 = GSL::Sf::sin(2*GSL::M_PI*freq1*t)
    x4.len.times do |index|
      if rand < 0.5
        x4[index] += rand
      end
    end
    y4 = x4.fft

    y1p = y1.subvector(1, n-2).to_complex2
    mag1 = y1p.abs
    phase1 = y1p.arg
    f1 = GSL::Vector.linspace(0, sampling/2, mag1.size)

    y2p = y2.subvector(1, n-2).to_complex2
    mag2 = y2p.abs
    phase2 = y2p.arg
    f2 = GSL::Vector.linspace(0, sampling/2, mag2.size)

    y3p = y3.subvector(1, n-2).to_complex2
    mag3 = y3p.abs
    phase3 = y3p.arg
    f3 = GSL::Vector.linspace(0, sampling/2, mag3.size)

    y4p = y4.subvector(1, n-2).to_complex2
    mag4 = y4p.abs
    phase4 = y4p.arg
    f4 = GSL::Vector.linspace(0, sampling/2, mag4.size)

    @plotter.make_plottable_data("fft_test",f1,mag1)
    @plotter.make_plot("-T png -C -g 3 -x 0 200 -X 'Frequency: unit amp'","fft_test","fft_plot.png")

    @plotter.make_plottable_data("fft_test2",f2,mag2)
    @plotter.make_plot("-T png -C -g 3 -x 0 200 -X 'Frequency: doubled amp'","fft_test2","fft_plot2.png")

    @plotter.make_plottable_data("fft_test3",f3,mag3)
    @plotter.make_plot("-T png -C -g 3 -x 0 200 -X 'Frequency: 2/3 amp'","fft_test3","fft_plot3.png")

    @plotter.make_plottable_data("fft_test4",f4,mag4)
    @plotter.make_plot("-T png -C -g 3 -x 0 200 -X 'Frequency: unit amp, noise added'","fft_test4","fft_plot4.png")
  end

end
