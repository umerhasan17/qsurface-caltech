# Correlation between soft output signal ϕ (Definition 9) and the log-likelihood ratio of
# pL = Pr(decoder returned a logical failure|σ) for individual physical qubit error rate p = 0.08 and surface code distance d,
# assuming syndrome measurement is perfect
from typing import List, Tuple, Dict, Callable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from qsurface.main import initialize, run


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
        max_iterations: int
) -> Tuple[np.ndarray, np.ndarray]:
    num_checks, num_vars = H.shape

    # Pre-calculating neighbor lists for speed
    N_v = [H[:, v].nonzero()[0] for v in range(num_vars)]  # Neighbors of var v
    N_c = [H[c, :].nonzero()[0] for c in range(num_checks)]  # Neighbors of check c

    # ... (LLR initialization, Eq. 2) ...
    c_hat = np.array(inner_decisions)
    pfail_values = np.array([pfail_model(phi) for phi in inner_phis])
    pfail_values = np.clip(pfail_values, 1e-6, 1.0 - 1e-6)
    
    # Compute LLRs: LLR = (1 - 2*c_hat) * log((1 - p_fail) / p_fail)
    # This gives strong positive LLR when c_hat=0 (no error) and p_fail is low
    # And strong negative LLR when c_hat=1 (error) and p_fail is high
    initial_llrs = (1 - 2 * c_hat) * np.log((1 - pfail_values) / pfail_values)
    
    # The LLRs are already properly scaled by the p_fail model
    # No additional boosting needed - the model parameters handle this

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
    return final_decisions, final_llrs

def example_pfail_model(phi: float) -> float:
    normalized_phi = phi / 25.0
    log_likelihood = 2.0 * normalized_phi - 3.0
    p_fail = 1.0 / (1.0 + np.exp(-log_likelihood))
    return np.clip(p_fail, 1e-4, 1.0 - 1e-4)


def hard_decode_outer(inner_decisions: List[int], H: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Hard decoding: simple majority vote for repetition code, or syndrome-based for general codes.
    For [3,1] repetition code, this is just majority vote.
    Returns (final_decisions, success).
    """
    inner_decisions = np.array(inner_decisions)
    
    # For repetition code, use majority vote
    if H.shape[0] == 2 and H.shape[1] == 3:  # [3,1] repetition code
        # Majority vote: if 2 or more blocks have error, output error
        num_errors = np.sum(inner_decisions)
        if num_errors >= 2:
            # All blocks have error
            final_decisions = np.ones(3, dtype=int)
        else:
            # No error (majority is correct)
            final_decisions = np.zeros(3, dtype=int)
    else:
        # For general codes, use syndrome-based hard decoding
        # Try to find a valid codeword that matches the syndrome
        syndrome = (H.dot(inner_decisions) % 2).astype(int)
        if np.all(syndrome == 0):
            # Syndrome satisfied, use original decisions
            final_decisions = inner_decisions.copy()
        else:
            # Syndrome not satisfied - try flipping bits to satisfy syndrome
            # Simple approach: flip the bit that appears in most unsatisfied checks
            final_decisions = inner_decisions.copy()
            for _ in range(10):  # Max iterations
                syndrome = (H.dot(final_decisions) % 2).astype(int)
                if np.all(syndrome == 0):
                    break
                # Find variable that appears in most unsatisfied checks
                unsatisfied_checks = np.where(syndrome == 1)[0]
                if len(unsatisfied_checks) == 0:
                    break
                # Count appearances in unsatisfied checks
                var_counts = np.zeros(H.shape[1])
                for c in unsatisfied_checks:
                    var_counts += H[c, :]
                # Flip the variable with highest count
                flip_var = np.argmax(var_counts)
                final_decisions[flip_var] = 1 - final_decisions[flip_var]
    
    success = np.all(final_decisions == 0)
    return final_decisions, success


def run_single_trial(
        p_physical: float,
        decoder_name: str,
        d_inner: int,
        H_outer: np.ndarray,
        pfail_model: Callable[[float], float],
        max_iterations: int
) -> Dict:
    """
    Run a single trial of hierarchical decoding.
    Returns a dictionary with results for both soft and hard decoding.
    """
    outer_code_size = H_outer.shape[1]
    
    # Get inner decoder results for each block
    inner_hard_decisions = []
    inner_phi_scores = []
    
    for i in range(outer_code_size):
        result_dict = get_phi_inner(p_physical, decoder_name, d_inner, N=1)
        c_hat = 1 - result_dict['no_error']  # 0 = no error, 1 = logical error
        phi = result_dict['phi']
        inner_hard_decisions.append(c_hat)
        inner_phi_scores.append(phi)
    
    # SOFT DECODING: Run outer decoder with soft priors
    soft_decisions, soft_llrs = decode_outer_code(
        inner_decisions=inner_hard_decisions,
        inner_phis=inner_phi_scores,
        pfail_model=pfail_model,
        H=H_outer,
        max_iterations=max_iterations
    )
    soft_success = np.all(soft_decisions == 0)
    
    # Calculate p_L from soft LLRs
    def log_prob_zero(llr):
        return -np.logaddexp(0, -llr)
    
    log_prob_correct = np.sum([log_prob_zero(llr) for llr in soft_llrs])
    p_L_soft = -np.expm1(log_prob_correct)
    p_L_soft = np.clip(p_L_soft, 1e-15, 1.0 - 1e-15)
    
    # HARD DECODING: Just use hard decisions
    hard_decisions, hard_success = hard_decode_outer(inner_hard_decisions, H_outer)
    
    # For hard decoding, p_L is just 1 if failed, 0 if succeeded (or we can estimate from error rate)
    p_L_hard = 0.0 if hard_success else 1.0
    
    return {
        'inner_decisions': inner_hard_decisions,
        'inner_phis': inner_phi_scores,
        'soft_decisions': soft_decisions,
        'soft_llrs': soft_llrs,
        'soft_success': soft_success,
        'p_L_soft': p_L_soft,
        'hard_decisions': hard_decisions,
        'hard_success': hard_success,
        'p_L_hard': p_L_hard,
        'inner_errors': sum(inner_hard_decisions)  # Count of inner block errors
    }


def collect_data(
        p_physical: float,
        decoder_name: str,
        d_inner: int,
        H_outer: np.ndarray,
        pfail_model: Callable[[float], float],
        max_iterations: int,
        n_trials: int
) -> Dict:
    """
    Collect statistics over multiple trials.
    Returns results for both soft and hard decoding.
    """
    results = {
        'soft_error_rate': [],
        'hard_error_rate': [],
        'soft_p_L': [],
        'hard_p_L': [],
        'inner_phis': []
    }
    
    print(f"Running {n_trials} trials for p={p_physical}, decoder={decoder_name}...")
    
    for trial in range(n_trials):
        if (trial + 1) % 100 == 0:
            print(f"  Trial {trial + 1}/{n_trials}")
        
        trial_result = run_single_trial(
            p_physical=p_physical,
            decoder_name=decoder_name,
            d_inner=d_inner,
            H_outer=H_outer,
            pfail_model=pfail_model,
            max_iterations=max_iterations
        )
        
        # Soft decoding error rate (1 if soft decoding failed)
        soft_error_rate = 1.0 if not trial_result['soft_success'] else 0.0
        results['soft_error_rate'].append(soft_error_rate)
        results['soft_p_L'].append(trial_result['p_L_soft'])
        
        # Hard decoding error rate (1 if hard decoding failed)
        hard_error_rate = 1.0 if not trial_result['hard_success'] else 0.0
        results['hard_error_rate'].append(hard_error_rate)
        results['hard_p_L'].append(trial_result['p_L_hard'])
        
        results['inner_phis'].extend(trial_result['inner_phis'])
    
    return results


def generate_publication_chart(
        decoder_name: str,
        decoder_data: Dict,
        output_path: str = None,
        figsize: Tuple[float, float] = (5, 4)
) -> None:
    """
    Generate a publication-quality chart comparing soft vs hard decoding.
    
    Parameters:
    -----------
    decoder_name : str
        Name of the decoder
    decoder_data : Dict with structure {p_physical: results_dict}
        Results from collect_data for different error rates
    output_path : str
        Path to save the figure (auto-generated if None)
    figsize : Tuple
        Figure size in inches
    """
    if output_path is None:
        output_path = f"soft_vs_hard_{decoder_name}.png"
    
    # Set publication-quality style
    matplotlib.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'font.serif': ['Times', 'Palatino', 'Computer Modern Roman'],
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 12,
        'lines.linewidth': 2.0,
        'lines.markersize': 7,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
    })
    
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
    
    p_physical_list = sorted(decoder_data.keys())
    soft_error_rates = []
    hard_error_rates = []
    
    for p_phys in p_physical_list:
        results = decoder_data[p_phys]
        # Use empirical error rates (fraction of failed trials)
        # This gives a fair comparison between soft and hard decoding
        soft_er = np.mean(results['soft_error_rate'])
        hard_er = np.mean(results['hard_error_rate'])
        soft_error_rates.append(soft_er)
        hard_error_rates.append(hard_er)
    
    # Plot soft vs hard
    ax.plot(
        p_physical_list, soft_error_rates,
        marker='o',
        color='#2ca02c',
        label='Soft decoding',
        linewidth=2.0,
        markersize=7,
        linestyle='-',
        alpha=0.9
    )
    ax.plot(
        p_physical_list, hard_error_rates,
        marker='s',
        color='#d62728',
        label='Hard decoding',
        linewidth=2.0,
        markersize=7,
        linestyle='--',
        alpha=0.9
    )
    
    # Format plot
    ax.set_xlabel('Physical Error Rate $p$', fontsize=11)
    ax.set_ylabel('Logical Error Rate $p_L$', fontsize=11)
    decoder_label_map = {
        'ufbfs': 'BFS-UFD',
        'unionfind': 'Union Find'
    }
    decoder_label = decoder_label_map.get(decoder_name, decoder_name.upper() if decoder_name else 'Decoder')
    ax.set_title(f'Soft vs Hard Decoding ({decoder_label})', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-5, top=2.0)  # Allow up to 2.0 to see values approaching 1
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.set_xlim(left=0, right=0.11)  # Focus on range 0.01 to 0.1
    
    plt.tight_layout()
    plt.savefig(output_path, format='png', bbox_inches='tight', dpi=300)
    print(f"Chart saved to {output_path}")
    plt.close()


def main(P_PHYSICAL, DECODER_NAME, MAX_ITERATIONS):
    """
    Main function for single trial (kept for backward compatibility).
    """
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
    final_decisions, final_llrs = decode_outer_code(
        inner_decisions=inner_hard_decisions,
        inner_phis=inner_phi_scores,
        pfail_model=example_pfail_model,
        H=H_outer_code,
        max_iterations=MAX_ITERATIONS
    )

    print("---------------------------------------")
    print(f"Final Hard Decisions: {final_decisions}")
    print(f"Final LLRs: {final_llrs}")

    # --- Calculate Outer Code (Hierarchical) p_L ---
    def log_prob_zero(llr):
        return -np.logaddexp(0, -llr)

    log_prob_correct = np.sum([log_prob_zero(llr) for llr in final_llrs])
    p_L_outer = -np.expm1(log_prob_correct)
    p_L_outer = np.clip(p_L_outer, 1e-15, 1.0 - 1e-15)

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
    # Configuration
    # Physical error rates from 10^-2 (0.01) to 10^-1 (0.1)
    P_BITFLIPS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    DECODERS = ["ufbfs", "unionfind"]  # Compare UFBFS and Union Find
    D_INNER = 5  # Inner code distance
    MAX_ITERATIONS = 1000
    N_TRIALS = 1000  # Number of trials per configuration
    
    # Define outer code (e.g., [3,1] repetition code)
    H_outer_code = np.array([
        [1, 1, 0],
        [0, 1, 1]
    ])
    
    # Collect data for all configurations
    all_data = {}
    
    for DECODER_NAME in DECODERS:
        all_data[DECODER_NAME] = {}
        
        for P_PHYSICAL in P_BITFLIPS:
            print(f"\n{'='*60}")
            print(f"Collecting data: p={P_PHYSICAL}, decoder={DECODER_NAME}")
            print(f"{'='*60}")
            
            results = collect_data(
                p_physical=P_PHYSICAL,
                decoder_name=DECODER_NAME,
                d_inner=D_INNER,
                H_outer=H_outer_code,
                pfail_model=example_pfail_model,
                max_iterations=MAX_ITERATIONS,
                n_trials=N_TRIALS
            )
            
            all_data[DECODER_NAME][P_PHYSICAL] = results
            
            # Print summary
            soft_er = np.mean(results['soft_p_L'])
            hard_er = np.mean(results['hard_p_L'])
            improvement = hard_er / (soft_er + 1e-10)
            print(f"\nSummary for p={P_PHYSICAL}:")
            print(f"  Soft decoding error rate: {soft_er:.6f}")
            print(f"  Hard decoding error rate: {hard_er:.6f}")
            print(f"  Improvement factor: {improvement:.2f}x")
    
    # Generate separate charts for each decoder
    print(f"\n{'='*60}")
    print("Generating publication charts...")
    print(f"{'='*60}")
    
    for DECODER_NAME in DECODERS:
        generate_publication_chart(
            decoder_name=DECODER_NAME,
            decoder_data=all_data[DECODER_NAME],
            output_path=f"soft_vs_hard_{DECODER_NAME}.png"
        )
    
    print("\nDone!")

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
