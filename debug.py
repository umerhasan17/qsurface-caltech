import time

import numpy as np

from qsurface.main import initialize, run

elapsed_times = []

for decoder_name in ["ufbfs", "unionfind"]:
    for i in range(100):
        np.random.seed(0)
        start = time.time()
        code, decoder = initialize((10, 10), "planar", decoder_name, enabled_errors=["pauli"], plotting=False,
                                   initial_states=(0, 0))
        # decoder.config["print_steps"] = True
        result = run(code, decoder, error_rates={"p_bitflip": 0.1}, decode_initial=False, iterations=1)
        elapsed = time.time() - start
        elapsed_times.append(elapsed)
        print(f"Iteration {i + 1}: Elapsed time: {elapsed:.2f} seconds")

    average_time = sum(elapsed_times) / len(elapsed_times)
    print(f"Average elapsed time over for {decoder_name=} 100 iterations: {average_time:.5f} seconds")

# code, decoder = initialize((10, 10), "planar", "unionfind", enabled_errors=["pauli"], plotting=False, initial_states=(0, 0))
# decoder.config["print_steps"] = True
# result1 = run(code, decoder, error_rates={"p_bitflip": 0.1}, decode_initial=False, iterations=1)
# print(result1)
# _, decoder_ufbfs = initialize((10, 10), "planar", "ufbfs", enabled_errors=["pauli"], plotting=False, initial_states=(0, 0))
# decoder_ufbfs.code.error_rates = {"p_bitflip": 0.1}
# decoder_ufbfs.config["print_steps"] = True
# result = run(code, decoder_ufbfs, error_rates={"p_bitflip": 0.1}, decode_initial=False, iterations=1)
# print(result)


# regular union find:
# soft failure rates: 0.0665 0.1355 0.268 0.421 0.625 0.7935 0.925 0.9755
# hard failure rates: 0.056 0.0785 0.296 0.2085 0.3365 0.4355 0.721 0.951


#
# [HARD] p_phys = 0.005
#   estimated p_block (p_L) = 0.023
#
# [HARD] p_phys = 0.007
#   estimated p_block (p_L) = 0.039
#
# [HARD] p_phys = 0.009
#   estimated p_block (p_L) = 0.041
#
# [HARD] p_phys = 0.011
#   estimated p_block (p_L) = 0.049
#
# [HARD] p_phys = 0.014
#   estimated p_block (p_L) = 0.065
#
# [HARD] p_phys = 0.016
#   estimated p_block (p_L) = 0.0745
#
# [HARD] p_phys = 0.018
#   estimated p_block (p_L) = 0.076
#
# [HARD] p_phys = 0.020
#   estimated p_block (p_L) = 0.0875
# failure rate hard info: [0.007  0.0715 0.1025 0.1875 0.6005 0.8305 0.8385 0.9625]
