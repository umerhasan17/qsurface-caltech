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
        max_iterations: int  # <-- This is from Algorithm 3, Line 6
) -> Tuple[np.ndarray, np.ndarray]:
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
    return final_decisions, final_llrs

def example_pfail_model(phi: float) -> float:
    """
    Example p_fail model: p_fail(phi) = sigmoid(slope * phi + intercept)
    This should be fitted from calibration data in practice.
    """
    log_likelihood = 0.8 * phi + 0.0  # (slope * phi + intercept)
    return 1.0 / (1.0 + np.exp(log_likelihood))


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
    Returns a dictionary with results.
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
    
    # Run outer decoder
    final_decisions, final_llrs = decode_outer_code(
        inner_decisions=inner_hard_decisions,
        inner_phis=inner_phi_scores,
        pfail_model=pfail_model,
        H=H_outer,
        max_iterations=max_iterations
    )
    
    # Check if outer decoding succeeded (all-zero codeword)
    outer_success = np.all(final_decisions == 0)
    
    # Calculate p_L from final LLRs
    def log_prob_zero(llr):
        return -np.logaddexp(0, -llr)
    
    log_prob_correct = np.sum([log_prob_zero(llr) for llr in final_llrs])
    p_L_outer = -np.expm1(log_prob_correct)
    p_L_outer = np.clip(p_L_outer, 1e-15, 1.0 - 1e-15)
    
    return {
        'inner_decisions': inner_hard_decisions,
        'inner_phis': inner_phi_scores,
        'outer_decisions': final_decisions,
        'outer_llrs': final_llrs,
        'outer_success': outer_success,
        'p_L_outer': p_L_outer,
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
    """
    results = {
        'inner_error_rate': [],
        'outer_error_rate': [],
        'inner_phis': [],
        'p_L_outer': []
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
        
        # Inner code error rate (fraction of blocks with logical errors)
        inner_error_rate = trial_result['inner_errors'] / len(trial_result['inner_decisions'])
        results['inner_error_rate'].append(inner_error_rate)
        
        # Outer code error rate (1 if outer decoding failed)
        outer_error_rate = 1.0 if not trial_result['outer_success'] else 0.0
        results['outer_error_rate'].append(outer_error_rate)
        
        results['inner_phis'].extend(trial_result['inner_phis'])
        results['p_L_outer'].append(trial_result['p_L_outer'])
    
    return results


def generate_publication_chart(
        data_dict: Dict[str, Dict],
        output_path: str = "hierarchical_decoding_results.pdf",
        figsize: Tuple[float, float] = (6, 4.5)
) -> None:
    """
    Generate a publication-quality chart comparing inner vs outer code performance.
    
    Parameters:
    -----------
    data_dict : Dict with structure {decoder_name: {p_physical: results_dict}}
        Results from collect_data for different decoders and error rates
    output_path : str
        Path to save the figure
    figsize : Tuple
        Figure size in inches
    """
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
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=300)
    
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (decoder_name, decoder_data) in enumerate(data_dict.items()):
        p_physical_list = sorted(decoder_data.keys())
        inner_error_rates = []
        outer_error_rates = []
        inner_error_stds = []
        outer_error_stds = []
        
        for p_phys in p_physical_list:
            results = decoder_data[p_phys]
            inner_error_rates.append(np.mean(results['inner_error_rate']))
            outer_error_rates.append(np.mean(results['outer_error_rate']))
            inner_error_stds.append(np.std(results['inner_error_rate']))
            outer_error_stds.append(np.std(results['outer_error_rate']))
        
        # Plot 1: Logical error rate vs physical error rate
        label = decoder_name.upper() if decoder_name else 'Decoder'
        ax1.errorbar(
            p_physical_list, inner_error_rates,
            yerr=inner_error_stds,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            label=f'Inner Code ({label})',
            capsize=3,
            capthick=1,
            linestyle='--',
            alpha=0.7
        )
        ax1.errorbar(
            p_physical_list, outer_error_rates,
            yerr=outer_error_stds,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            label=f'Outer Code ({label})',
            capsize=3,
            capthick=1,
            linestyle='-',
            alpha=1.0
        )
        
        # Plot 2: Improvement factor (inner_error_rate / outer_error_rate)
        improvement = np.array(inner_error_rates) / (np.array(outer_error_rates) + 1e-10)
        ax2.plot(
            p_physical_list, improvement,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            label=label,
            linewidth=1.5
        )
    
    # Format Plot 1
    ax1.set_xlabel('Physical Error Rate $p$', fontsize=11)
    ax1.set_ylabel('Logical Error Rate $p_L$', fontsize=11)
    ax1.set_title('Hierarchical Decoding Performance', fontsize=12, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax1.set_xlim(left=0)
    
    # Format Plot 2
    ax2.set_xlabel('Physical Error Rate $p$', fontsize=11)
    ax2.set_ylabel('Improvement Factor', fontsize=11)
    ax2.set_title('Outer Code Improvement', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax2.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
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
    P_BITFLIPS = [0.06, 0.08, 0.1, 0.12, 0.16, 0.2]
    DECODERS = ["ufbfs"]  # Can add: "mwpm", "unionfind"
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
            inner_er = np.mean(results['inner_error_rate'])
            outer_er = np.mean(results['outer_error_rate'])
            improvement = inner_er / (outer_er + 1e-10)
            print(f"\nSummary for p={P_PHYSICAL}:")
            print(f"  Inner code error rate: {inner_er:.4f}")
            print(f"  Outer code error rate: {outer_er:.4f}")
            print(f"  Improvement factor: {improvement:.2f}x")
    
    # Generate publication chart
    print(f"\n{'='*60}")
    print("Generating publication chart...")
    print(f"{'='*60}")
    generate_publication_chart(
        data_dict=all_data,
        output_path="hierarchical_decoding_results.pdf"
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
