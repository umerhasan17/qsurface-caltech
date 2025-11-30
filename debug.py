from qsurface.main import initialize, run
import numpy as np
np.random.seed(0)
code, decoder = initialize((10,10), "planar", "ufbfs", enabled_errors=["pauli"], plotting=False, initial_states=(0,0))
result = run(code, decoder, error_rates = {"p_bitflip": 0.1}, decode_initial=False, iterations=1)
print(result)