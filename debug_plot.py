# for decoder in ['mwpm', 'unionfind', 'ufns']:
#     for code_type in ['planar']:
#         for size in [(4,4), (6,6), (8,8)]:
#             print(f"Testing decoder: {decoder}, code type: {code_type}, size: {size}")
#             try:
#                 from qsurface.main import initialize, run
#                 code, decoder_instance = initialize(size, code_type, decoder, enabled_errors=["pauli"], plotting=True, initial_states=(0,0))
#                 result = run(code, decoder_instance, error_rates = {"p_bitflip": 0.1}, decode_initial=False, iterations=1)
#                 print(f"Result: {result}")
#             except Exception as e:
#                 print(f"Error occurred with {decoder=}, {code_type=}, {size=}: {e=}")
#                 raise e

# from qsurface.main import initialize, run
# code, decoder = initialize((6,6), "toric", "mwpm", enabled_errors=["pauli"], plotting=True, initial_states=(0,0))
# result = run(code, decoder, error_rates = {"p_bitflip": 0.1, "p_phaseflip": 0.1}, decode_initial=False)
# print(result)

from qsurface.main import initialize, run
code, decoder = initialize((6,6), "planar", "unionfind", enabled_errors=["pauli"], plotting=False, initial_states=(0,0))
res = run(code, decoder, error_rates = {"p_bitflip": 0.1, "p_phaseflip": 0.1}, decode_initial=False)
print(res)
# TODO(uhasan) remove phase flip