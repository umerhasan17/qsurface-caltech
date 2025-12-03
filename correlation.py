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

    decoders = ["unionfind", "ufbfs"]

    for decoder_name in decoders:
        for p_physical in [0.06, 0.1, 0.12, 0.16]:
            avg_phi, log_likelihood_per_bin, counts = main_inner_code(P_PHYSICAL=p_physical, DECODER_NAME=decoder_name,
                                                                      MAX_ITERATIONS=100)
            results_dict[(decoder_name, p_physical)] = [avg_phi, log_likelihood_per_bin, counts]

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    # --- 1. SETUP COLORS (Define once) ---
    # Dark Navy (#061A40) -> Turquoise (#40E0D0)
    cols = ["#061A40", "#40E0D0"]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("navy_turquoise", cols)

    # Define the absolute range for normalization
    norm = mcolors.Normalize(vmin=0.06, vmax=0.16)

    # --- 2. MAIN PLOTTING LOOP ---
    for decoder_name in decoders:

        # Create a new figure and axis for THIS decoder
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get all p values available for this specific decoder
        p_physical_values = sorted(set([p for (d, p) in results_dict.keys() if d == decoder_name]))

        for p_physical in p_physical_values:
            # Unpack data
            avg_phi, log_likelihood_per_bin, counts = results_dict[(decoder_name, p_physical)]

            # Filter out invalid values (inf, nan)
            valid_mask = np.isfinite(avg_phi) & np.isfinite(log_likelihood_per_bin)
            avg_phi_valid = avg_phi[valid_mask]
            log_likelihood_valid = log_likelihood_per_bin[valid_mask]

            if len(avg_phi_valid) == 0:
                continue

            # Get color based on p_physical
            current_color = custom_cmap(norm(p_physical))

            # A. Plot Data Points (Add label here for the legend)
            ax.plot(avg_phi_valid, log_likelihood_valid,
                    marker='o',
                    linestyle='none',
                    color=current_color,
                    label=f'$p={p_physical}$')

            # B. Fit and Plot Line (No label here to keep legend clean)
            if len(avg_phi_valid) > 1:
                coeffs = np.polyfit(avg_phi_valid, log_likelihood_valid, 1)
                phi_fit = np.linspace(avg_phi_valid.min(), avg_phi_valid.max(), 100)
                log_likelihood_fit = np.polyval(coeffs, phi_fit)

                ax.plot(phi_fit, log_likelihood_fit,
                        linestyle='--',
                        linewidth=1.5,
                        alpha=0.7,
                        color=current_color)

        # --- 3. FORMATTING & LABELS ---
        ax.set_xlabel(r"$\phi$ (binned)", fontsize=12)
        ax.set_ylabel("Log Likelihood Logical Error Rate", fontsize=12)
        ax.set_title(f"Correlation for Decoder: {decoder_name}", fontsize=14)

        ax.grid(True, alpha=0.3)

        # FORCE LEGEND TO TOP LEFT
        ax.legend(title="Physical Error Rate", loc='upper left')

        # Save and Show
        plt.tight_layout()
        plt.savefig(f"correlation_{decoder_name}.pdf")
        plt.show()

        # Close figure to clear memory
        plt.close(fig)