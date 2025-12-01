# Correlation between soft output signal ϕ (Definition 9) and the log-likelihood ratio of
# pL = Pr(decoder returned a logical failure|σ) for individual physical qubit error rate p = 0.08 and surface code distance d,
# assuming syndrome measurement is perfect
from typing import Dict

import numpy as np



def get_phi_inner(p_bitflip: float, decoder: str, d: int, N: int) -> Dict:
    from qsurface.main import initialize, run
    code, decoder_instance = initialize((d, d), "planar", decoder, enabled_errors=["pauli"], plotting=False,
                                        initial_states=(0, 0))
    result = run(code, decoder_instance, error_rates={"p_bitflip": p_bitflip}, decode_initial=False, iterations=N)
    return result


def estimate_pL_vs_phi(phis, Ls, bin_edges):
    """
    Estimate logical error probability p_L as a function of soft output phi
    by binning data in phi.

    Parameters
    ----------
    phis : array-like of shape (N,)
        Soft outputs for each trial.
    Ls : array-like of shape (N,)
        Logical failure indicators for each trial (0 = success, 1 = failure).
    bin_edges : array-like of shape (M+1,)
        Bin edges for phi: [b0, b1, ..., bM]. Bins are [b_k, b_{k+1}).

    Returns
    -------
    avg_phi_per_bin : np.ndarray
        Mean phi value for samples in each non-empty bin.
    pL_per_bin : np.ndarray
        Estimated logical error probability in each non-empty bin.
    counts_per_bin : np.ndarray
        Number of samples in each non-empty bin.
    """

    phis = np.asarray(phis, dtype=float)
    Ls = np.asarray(Ls, dtype=float)
    bin_edges = np.asarray(bin_edges, dtype=float)

    assert phis.shape == Ls.shape, "phis and Ls must have the same length"
    num_bins = len(bin_edges) - 1

    # digitize: returns bin index in 1..num_bins (inclusive) for each phi
    # where bin k is [bin_edges[k-1], bin_edges[k])
    bin_indices = np.digitize(phis, bin_edges) - 1  # shift to 0..num_bins-1

    # Initialize accumulators
    sum_phi_in_bin = np.zeros(num_bins, dtype=float)
    sum_L_in_bin = np.zeros(num_bins, dtype=float)
    count_in_bin = np.zeros(num_bins, dtype=int)

    # Accumulate stats per bin
    for phi, L, k in zip(phis, Ls, bin_indices):
        if 0 <= k < num_bins:
            sum_phi_in_bin[k] += phi
            sum_L_in_bin[k] += L
            count_in_bin[k] += 1
        # else: phi fell outside the specified bin range → ignore

    # Compute averages for non-empty bins
    nonempty = count_in_bin > 0
    avg_phi_per_bin = sum_phi_in_bin[nonempty] / count_in_bin[nonempty]
    pL_per_bin = sum_L_in_bin[nonempty] / count_in_bin[nonempty]
    log_likelihood_per_bin = np.log((1.0 - pL_per_bin) / pL_per_bin)

    counts_per_bin = count_in_bin[nonempty]

    return avg_phi_per_bin, log_likelihood_per_bin, counts_per_bin


def main_inner_code(P_PHYSICAL, DECODER_NAME, MAX_ITERATIONS):
    """
    Main function for single trial (kept for backward compatibility).
    """
    # 2. DEFINE INNER CODE PARAMETERS
    D_INNER = 10
    N_INNER = 1

    # 3. GET INNER DECODER RESULTS
    inner_hard_decisions = []
    inner_phi_scores = []

    for i in range(MAX_ITERATIONS):
        # Call your API for each block of the outer code
        result_dict = get_phi_inner(P_PHYSICAL, DECODER_NAME, D_INNER, N_INNER)

        # 'no_error' = 1 means decoder decided 0 (no logical error)
        # 'no_error' = 0 means decoder decided 1 (logical error)
        c_hat = 1 - result_dict['no_error']
        phi = result_dict['phi']

        inner_hard_decisions.append(c_hat)
        inner_phi_scores.append(phi)

    avg_phi, log_likelihood_per_bin, counts = estimate_pL_vs_phi(phis=inner_phi_scores, Ls=inner_hard_decisions,
                                                                 bin_edges=np.linspace(0, 20, 20))

    # log_prob_correct = np.sum([log_prob_zero(llr) for llr in final_llrs])
    # p_L_outer = -np.expm1(log_prob_correct)
    # p_L_outer = np.clip(p_L_outer, 1e-15, 1.0 - 1e-15)
    #
    # log_likelihood_of_failure = np.log((1.0 - p_L_outer) / p_L_outer)
    return avg_phi, log_likelihood_per_bin, counts
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
    np.random.seed(42)

    results_dict = {}

    decoders = ["ufbfs"]

    for decoder_name in decoders:
        for p_physical in [0.06, 0.08, 0.1, 0.12, 0.16, 0.2]:
            avg_phi, log_likelihood_per_bin, counts = main_inner_code(P_PHYSICAL=p_physical, DECODER_NAME=decoder_name,
                                                                      MAX_ITERATIONS=2_000)
            results_dict[(decoder_name, p_physical)] = [avg_phi, log_likelihood_per_bin, counts]

    import matplotlib.pyplot as plt

    # ufbfs
    for decoder_name in decoders:
        # Get all unique p_physical values for this decoder
        p_physical_values = sorted(set([p for (d, p) in results_dict.keys() if d == decoder_name]))

        # Plot one line per p_physical
        for p_physical in p_physical_values:
            avg_phi, log_likelihood_per_bin, counts = results_dict[(decoder_name, p_physical)]

            # Filter out invalid values (inf, nan)
            valid_mask = np.isfinite(avg_phi) & np.isfinite(log_likelihood_per_bin)
            avg_phi_valid = avg_phi[valid_mask]
            log_likelihood_valid = log_likelihood_per_bin[valid_mask]

            # Plot data points
            line_handle = plt.plot(avg_phi_valid, log_likelihood_valid, marker='o', linestyle='none',
                                   label=f'p={p_physical}')
            point_color = line_handle[0].get_color()

            # Fit a line (linear regression)
            if len(avg_phi_valid) > 1:
                # Fit polynomial of degree 1 (straight line)
                coeffs = np.polyfit(avg_phi_valid, log_likelihood_valid, 1)
                # Generate points for the fitted line
                phi_fit = np.linspace(avg_phi_valid.min(), avg_phi_valid.max(), 100)
                log_likelihood_fit = np.polyval(coeffs, phi_fit)
                # Plot the fitted line
                plt.plot(phi_fit, log_likelihood_fit, linestyle='--', alpha=0.7, linewidth=1.5, color=point_color)

        plt.xlabel("soft output phi (binned)")
        plt.ylabel("log likelihood logical error rate p_L")
        plt.title(f"Correlation for decoder: {decoder_name}")
        plt.legend()
        plt.show()
        # main_inner_code(P_PHYSICAL=0.1, DECODER_NAME="ufbfs", MAX_ITERATIONS=1_000)
        # main_inner_code(P_PHYSICAL=0.1, DECODER_NAME="unionfind", MAX_ITERATIONS=1_000)
