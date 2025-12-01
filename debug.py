from qsurface.main import initialize, run
import numpy as np

for decoder_name in ["unionfind", "ufbfs"]:
    np.random.seed(0)
    code, decoder = initialize((10, 10), "planar", decoder_name, enabled_errors=["pauli"], plotting=False,
                               initial_states=(0, 0))
    decoder.config["print_steps"] = True
    result = run(code, decoder, error_rates={"p_bitflip": 0.1}, decode_initial=False, iterations=1)
    print(result)


# code, decoder = initialize((10, 10), "planar", "unionfind", enabled_errors=["pauli"], plotting=False, initial_states=(0, 0))
# decoder.config["print_steps"] = True
# result1 = run(code, decoder, error_rates={"p_bitflip": 0.1}, decode_initial=False, iterations=1)
# print(result1)
# _, decoder_ufbfs = initialize((10, 10), "planar", "ufbfs", enabled_errors=["pauli"], plotting=False, initial_states=(0, 0))
# decoder_ufbfs.code.error_rates = {"p_bitflip": 0.1}
# decoder_ufbfs.config["print_steps"] = True
# result = run(code, decoder_ufbfs, error_rates={"p_bitflip": 0.1}, decode_initial=False, iterations=1)
# print(result)