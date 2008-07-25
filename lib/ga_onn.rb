# Error weighting function. Calculates a Euclidean-style weighted error measure of the output.
    #   expected: the expected/desired result (GSL::Matrix of data)
        def weighted_error(expected)
              w_err = 0.0
                    result_amps = []
                          result_freqs = []
                                @nodes.size-@num_outputs..@nodes.size do |index|
                                        result_amps_i, result_freqs_i = fourier_analyze(index)
                                                result_ampss << result_amps_i
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

