# Correlation between soft output signal ϕ (Definition 9) and the log-likelihood ratio of
# pL = Pr(decoder returned a logical failure|σ) for individual physical qubit error rate p = 0.08 and surface code distance d,
# assuming syndrome measurement is perfect
from typing import List

import numpy as np
import math
from qsurface.main import initialize, run
import numpy as np
from typing import List, Callable, Dict


def get_phi_inner(p_bitflip: float, decoder: str, d: int, N: int) -> float:
    code, decoder_instance = initialize((d, d), "planar", decoder, enabled_errors=["pauli"], plotting=False,
                                        initial_states=(0, 0))
    result = run(code, decoder_instance, error_rates={"p_bitflip": p_bitflip}, decode_initial=False, iterations=N)
    return result


def decode_outer_code(
        inner_decisions: List[int],
        inner_phis: List[float],
        pfail_model: Callable[[float], float],
        H: np.ndarray,
        max_iterations: int  # <-- This is from Algorithm 3, Line 6
) -> np.ndarray:
    num_checks, num_vars = H.shape

    # Pre-calculating neighbor lists for speed
    N_v = [H[:, v].nonzero()[0] for v in range(num_vars)]  # Neighbors of var v
    N_c = [H[c, :].nonzero()[0] for c in range(num_checks)]  # Neighbors of check c

    # ... (LLR initialization, Eq. 2) ...
    c_hat = np.array(inner_decisions)
    pfail_values = np.array([pfail_model(phi) for phi in inner_phis])
    pfail_values = np.clip(pfail_values, 1e-15, 1.0 - 1e-15)
    initial_llrs = (1 - 2 * c_hat) * np.log((1 - pfail_values) / pfail_values)

    m_v_to_c = np.zeros((num_checks, num_vars))
    m_c_to_v = np.zeros((num_checks, num_vars))

    # --- Algorithm 3, Line 3: "repeat ... until max_iterations" ---
    for _ in range(max_iterations):

        # --- Algorithm 3, Line 4: Update check-to-variable (Eq. 4) ---
        # This loop uses your N_c list
        for c in range(num_checks):
            for v in N_c[c]:  # <-- Using N_c[c] to get all v in N(c)
                product_term = 1.0
                for v_prime in N_c[c]:  # <-- Using N_c[c] again
                    if v_prime == v:
                        continue
                    product_term *= np.tanh(m_v_to_c[c, v_prime] / 2.0)

                product_term = np.clip(product_term, -0.9999999, 0.9999999)
                m_c_to_v[c, v] = 2.0 * np.arctanh(product_term)

        # --- Algorithm 3, Line 5: Update variable-to-check (Eq. 3) ---
        # This loop uses the N_v list
        for v in range(num_vars):
            for c in N_v[v]:
                sum_term = 0.0
                for c_prime in N_v[v]:
                    if c_prime == c:
                        continue
                    sum_term += m_c_to_v[c_prime, v]

                m_v_to_c[c, v] = initial_llrs[v] + sum_term

        # --- Algorithm 3, Line 6: Check for early stopping ---
        final_llrs_t = np.copy(initial_llrs)
        for v in range(num_vars):
            final_llrs_t[v] += np.sum(m_c_to_v[:, v])
        final_decisions_t = (final_llrs_t < 0).astype(int)

        syndrome = H.dot(final_decisions_t) % 2
        if np.all(syndrome == 0):
            break  # Syndrome satisfied, stop early.

    # ... (Final output calculation) ...
    final_llrs = np.copy(initial_llrs)
    for v in range(num_vars):
        final_llrs[v] += np.sum(m_c_to_v[:, v])

    final_decisions = (final_llrs < 0).astype(int)
    return final_decisions

def example_pfail_model(phi: float) -> float:
    log_likelihood = 0.8 * phi + 0.0  # (slope * phi + intercept)
    return 1.0 / (1.0 + np.exp(log_likelihood))


def main(P_PHYSICAL, DECODER_NAME, MAX_ITERATIONS):
    # 1. DEFINE THE OUTER CODE (e.g., a [3,1] Repetition Code)
    H_outer_code = np.array([
        [1, 1, 0],
        [0, 1, 1]
    ])
    OUTER_CODE_SIZE = H_outer_code.shape[1]  # num_vars = 3

    # 2. DEFINE INNER CODE PARAMETERS
    D_INNER = 5
    N_INNER = 1

    # 3. GET INNER DECODER RESULTS
    inner_hard_decisions = []
    inner_phi_scores = []

    print(f"Running hierarchical simulation for {OUTER_CODE_SIZE} inner blocks...")

    for i in range(OUTER_CODE_SIZE):
        # Call your API for each block of the outer code
        result_dict = get_phi_inner(P_PHYSICAL, DECODER_NAME, D_INNER, N_INNER)

        # 'no_error' = 1 means decoder decided 0 (no logical error)
        # 'no_error' = 0 means decoder decided 1 (logical error)
        c_hat = 1 - result_dict['no_error']
        phi = result_dict['phi']

        inner_hard_decisions.append(c_hat)
        inner_phi_scores.append(phi)

        print(f"Block {i}: Hard Decision (ĉ)={c_hat}, Phi (ϕ)={phi:.2f}")

    # 4. RUN THE OUTER DECODER
    # Assumes decode_outer_code() and example_pfail_model() exist
    final_code_word = decode_outer_code(
        inner_decisions=inner_hard_decisions,
        inner_phis=inner_phi_scores,
        pfail_model=example_pfail_model,
        H=H_outer_code,
        max_iterations=MAX_ITERATIONS
    )

    print("---------------------------------------")
    print(f"Final Variable LLRs (Beliefs): {final_code_word}")

    # --- Calculate Outer Code (Hierarchical) p_L ---
    # p_L_outer = 1 - P(word == [0,0,0])
    # P(v=0) = 1 / (1 + exp(-LLR))
    # We use log-space calculations for numerical stability

    def log_prob_zero(llrs):
        return -np.logaddexp(0, -llrs)

    # log( P(word == [0,0,0]) ) = sum( log(P(v=0)) )
    log_prob_correct = np.sum(log_prob_zero(final_code_word))

    # p_L_outer = 1 - P(correct) = 1 - exp(log_prob_correct)
    # Use -expm1(x) which is a stable way to compute 1 - exp(x)
    p_L_outer = -np.expm1(log_prob_correct)

    # Clamp to avoid log(0) if p_L is 0 or 1
    p_L_outer = np.clip(p_L_outer, 1e-15, 1.0 - 1e-15)

    # --- Calculate the requested Log-Likelihood of Failure ---
    log_likelihood_of_failure = np.log((1.0 - p_L_outer) / p_L_outer)

    print(f"Hierarchical (Outer) p_L: {p_L_outer:.4e}")
    print(f"Hierarchical Log-Likelihood (log(1-p_L)/p_L): {log_likelihood_of_failure:.4f}")

    return log_likelihood_of_failure


    #
    # print("---------------------------------------")
    # print(f"Final Corrected Codeword: {final_code_word}")
    #
    # # Check if the final word is the all-zero (correct) word
    # if np.all(final_code_word == 0):
    #     print("Result: Hierarchical decoding SUCCESSFUL.")
    # else:
    #     print("Result: Hierarchical decoding FAILED.")


if __name__ == "__main__":
    # N_TRIALS = 100_000
    P_BITFLIPS = [0.06, 0.08, 0.1, 0.12, 0.16, 0.2]
    DECODERS = ["ufbfs"]  # "mwpm", "unionfind",
    D = 5  # Inner code distance
    OUTER_CODE_SIZE = 3  # A simple [3,1] repetition code
    MAX_ITERATIONS = 1_000
    # P_PHYSICAL = 0.1
    # DECODER_NAME = "ufbfs"

    for P_PHYSICAL in P_BITFLIPS:
        for DECODER_NAME in DECODERS:
            print(f"\n=== Running hierarchical simulation with p_physical={P_PHYSICAL}, decoder={DECODER_NAME} ===")
            main(P_PHYSICAL=P_PHYSICAL, DECODER_NAME=DECODER_NAME, MAX_ITERATIONS=MAX_ITERATIONS)

    # for p_bitflip in P_BITFLIPS:
    #     for decoder in DECODERS:
    #         phis, logical_error_rates = run_hierarchical_simulation(N_TRIALS, p_bitflip, OUTER_CODE_SIZE)

    # N = 1_000
    # size = (5, 5)
    # results_array = []
    #
    # for decoder in ["mwpm", "unionfind", "ufbfs"]:
    #     for p_bitflip in [0.06, 0.08, 0.1, 0.12, 0.16, 0.2]:
    #         print(f"Testing decoder: {decoder}, p_bitflip: {p_bitflip}")
    #         try:
    #             from qsurface.main import initialize, run
    #
    #             code, decoder_instance = initialize(size, "planar", decoder, enabled_errors=["pauli"], plotting=False,
    #                                                 initial_states=(0, 0))
    #             result = run(code, decoder_instance, error_rates={"p_bitflip": p_bitflip}, decode_initial=False,
    #                          iterations=N)
    #             print(f"Result: {result}")
    #         except Exception as e:
    #             print(f"Error occurred with {decoder=}, p_bitflip={p_bitflip}: {e=}")
    #             raise e
