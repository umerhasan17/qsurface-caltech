import numpy as np
from qldpc.abstract import CyclicGroup, GroupRing, RingArray, RingMember
from qldpc.codes import LPCode
from qldpc.objects import Pauli
import matplotlib.pyplot as plt
import correlation
from tqdm import tqdm


def build_qc_lifted_product_code(lift_size: int = 31) -> LPCode:
    """
    Quasi-cyclic lifted product code from Appendix C:

      - lift size ℓ = 31
      - base matrix A over F2[x]/(x^ℓ - 1):

            ( x     x^2    x^4    x^8    x^16
              x^5   x^10   x^20   x^9    x^18
              x^25  x^19   x^7    x^14   x^28 )

    Returns the corresponding LPCode instance.
    """

    # 1. Group and group ring: F2[Z_ℓ]
    group = CyclicGroup(lift_size)      # Z_ℓ
    ring = GroupRing(group)            # F2[Z_ℓ]
    gen = group.generators[0]          # generator of Z_ℓ
    x = RingMember(ring, gen)          # “x” in the appendix

    # 2. Exponents of x in the base matrix A (Eq. C3)
    exponents = [
        [1, 2, 4, 8, 16],
        [5, 10, 20, 9, 18],
        [25, 19, 7, 14, 28],
    ]

    # 3. Build the RingArray for A: entries are x^k in F2[Z_ℓ]
    A_entries = [[x ** e for e in row] for row in exponents]
    A = RingArray.build(A_entries, ring)

    # 4. Lifted-product code:
    #    LPCode(matrix_a, matrix_b=None) defaults to matrix_b = matrix_a,
    #    which matches the “lifted product of a single base matrix” setup.
    code = LPCode(A)

    # 5. Sanity checks on parameters (should match the appendix text)
    print("Number of physical qubits n:", code.num_qudits)
    print("Number of logical qubits k:", code.dimension)

    # Expected from the construction: n = ℓ * (n^2 + m^2) with m=3, n=5, ℓ=31
    expected_n = lift_size * (5 ** 2 + 3 ** 2)
    print("Expected n from formula:", expected_n)

    # For convenience, you can assert these if you like:
    # assert code.num_qudits == expected_n
    # assert code.dimension == 140

    return code


def get_Hz_matrix(code: LPCode) -> np.ndarray:
    """
    Return H_z (Z-type parity-check matrix) as a numpy int array of shape (m, n).
    H_z witnesses X (bit-flip) errors.
    """
    # matrix_z is a galois FieldArray; cast down to ints mod 2
    Hz = np.array(code.matrix_z, dtype=int) % 2
    return Hz



def bp_decode_bsc(H: np.ndarray,
                  syndrome: np.ndarray,
                  p: float,
                  max_iterations: int = 50
                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Standard belief-propagation decoder on a binary Tanner graph (LDPC code).

    Parameters
    ----------
    H : (m, n) ndarray of {0,1}
        Parity-check matrix.
    syndrome : (m,) ndarray of {0,1}
        Measured syndrome s = H e (mod 2).
    p : float
        Bit-flip probability of the BSC channel.
    max_iterations : int
        Maximum number of BP iterations.

    Returns
    -------
    est_error : (n,) ndarray of {0,1}
        Estimated error pattern.
    final_llrs : (n,) ndarray of floats
        Final log-likelihood ratios for each variable node.
    """
    H = H.astype(int)
    syndrome = syndrome.astype(int)
    num_checks, num_vars = H.shape

    # Neighbour lists for Tanner graph
    N_v = [np.where(H[:, v] == 1)[0] for v in range(num_vars)]   # checks touching v
    N_c = [np.where(H[c, :] == 1)[0] for c in range(num_checks)] # vars touching c

    # Channel LLR for BSC(p): ln( (1-p) / p )
    L_ch = np.log((1.0 - p) / p)

    # Messages (check <-> variable) on each edge.
    # We only ever use entries where H[c, v] == 1.
    m_v_to_c = np.zeros((num_checks, num_vars), dtype=float)
    m_c_to_v = np.zeros((num_checks, num_vars), dtype=float)

    # Initialize variable-to-check messages with channel LLR
    for c in range(num_checks):
        for v in N_c[c]:
            m_v_to_c[c, v] = L_ch

    eps = 1e-12

    for _ in range(max_iterations):
        # ----- Check-to-variable update: Eq. (E2) -----
        for c in range(num_checks):
            # syndrome bit s_c gives the overall sign factor (-1)^{s_c}
            sign_c = -1.0 if syndrome[c] == 1 else 1.0

            for v in N_c[c]:
                prod = 1.0
                for v_prime in N_c[c]:
                    if v_prime == v:
                        continue
                    prod *= np.tanh(m_v_to_c[c, v_prime] / 2.0)

                # Numerical clipping to keep atanh well-defined
                prod = np.clip(prod, -1.0 + eps, 1.0 - eps)
                m_c_to_v[c, v] = sign_c * 2.0 * np.arctanh(prod)

        # ----- Variable-to-check update: Eq. (E1) -----
        for v in range(num_vars):
            for c in N_v[v]:
                sum_term = L_ch  # ln((1-p)/p)
                for c_prime in N_v[v]:
                    if c_prime == c:
                        continue
                    sum_term += m_c_to_v[c_prime, v]

                m_v_to_c[c, v] = sum_term

        # ----- Form tentative decision & early stopping -----
        final_llrs = np.full(num_vars, L_ch, dtype=float)
        for v in range(num_vars):
            for c in N_v[v]:
                final_llrs[v] += m_c_to_v[c, v]

        est_error = (final_llrs < 0).astype(int)  # bit = 1 if LLR < 0

        if np.all((H @ est_error) % 2 == syndrome):
            break

    return est_error, final_llrs


def bp_decode_outer_lp_hard(code: LPCode,
                       syndrome: np.ndarray,
                       p_block: float,
                       max_iterations: int = 50
                      ) -> tuple[np.ndarray, np.ndarray]:
    """
    Belief-propagation decoder for the lifted-product outer code, assuming
    each outer qubit experiences an independent bit flip with prob. p_block.

    Uses H_z from the CSS code and decodes X errors.
    """
    Hz = get_Hz_matrix(code)
    return bp_decode_bsc(Hz, syndrome, p_block, max_iterations=max_iterations)


def hard_only_llrs(inner_decisions, base_llr: float = 2.0) -> np.ndarray:
    """
    Use only the hard inner decisions c_hat to build priors.
    c_hat = 0 -> positive LLR (no error)
    c_hat = 1 -> negative LLR (error)
    """
    c_hat = np.array(inner_decisions, dtype=int)
    return (1 - 2 * c_hat) * base_llr


def bp_decode_outer_hard(code: LPCode,
                         inner_decisions,
                         syndrome,
                         max_iterations: int = 50):
    """
    Outer BP decoder that ignores φ and only uses hard inner decisions.
    """
    Hz = get_Hz_matrix(code)
    prior_llrs = hard_only_llrs(inner_decisions)
    return bp_decode_with_priors(Hz, prior_llrs, syndrome, max_iterations=max_iterations)



def bp_decode_with_priors(H: np.ndarray,
                          prior_llrs: np.ndarray,
                          syndrome: np.ndarray,
                          max_iterations: int = 50
                         ) -> tuple[np.ndarray, np.ndarray]:
    """
    Belief-propagation decoder on Tanner graph defined by H,
    using *arbitrary per-variable prior LLRs* (one per column of H)
    and an arbitrary target syndrome.

    Parameters
    ----------
    H : (m, n) ndarray
        Parity-check matrix over F2.
    prior_llrs : (n,) ndarray
        Initial LLRs for each variable node.
    syndrome : (m,) ndarray
        Target syndrome s, so we enforce H e_hat = s (mod 2).
    max_iterations : int
        BP iterations.

    Returns
    -------
    est_error : (n,) ndarray of {0,1}
        Estimated error pattern.
    final_llrs : (n,) ndarray
        Final LLRs.
    """
    H = H.astype(int)
    syndrome = syndrome.astype(int)
    num_checks, num_vars = H.shape

    # Neighbour lists
    N_v = [np.where(H[:, v] == 1)[0] for v in range(num_vars)]
    N_c = [np.where(H[c, :] == 1)[0] for c in range(num_checks)]

    # Messages
    m_v_to_c = np.zeros((num_checks, num_vars))
    m_c_to_v = np.zeros((num_checks, num_vars))

    # Initialise variable-to-check messages with priors
    for v in range(num_vars):
        for c in N_v[v]:
            m_v_to_c[c, v] = prior_llrs[v]

    eps = 1e-12

    for _ in range(max_iterations):
        # --- check -> variable updates (with syndrome sign) ---
        for c in range(num_checks):
            sign_c = -1.0 if syndrome[c] == 1 else 1.0

            for v in N_c[c]:
                prod = 1.0
                for v_prime in N_c[c]:
                    if v_prime == v:
                        continue
                    prod *= np.tanh(m_v_to_c[c, v_prime] / 2.0)

                prod = np.clip(prod, -1.0 + eps, 1.0 - eps)
                m_c_to_v[c, v] = sign_c * 2.0 * np.arctanh(prod)

        # --- variable -> check updates ---
        for v in range(num_vars):
            for c in N_v[v]:
                sum_term = prior_llrs[v]
                for c_prime in N_v[v]:
                    if c_prime == c:
                        continue
                    sum_term += m_c_to_v[c_prime, v]
                m_v_to_c[c, v] = sum_term

        # --- tentative decision for early stopping ---
        final_llrs_t = np.copy(prior_llrs)
        for v in range(num_vars):
            final_llrs_t[v] += np.sum(m_c_to_v[:, v])
        est_t = (final_llrs_t < 0).astype(int)

        if np.all((H @ est_t) % 2 == syndrome):
            break

    # Final marginals
    final_llrs = np.copy(prior_llrs)
    for v in range(num_vars):
        final_llrs[v] += np.sum(m_c_to_v[:, v])

    est_error = (final_llrs < 0).astype(int)
    return est_error, final_llrs


def phi_to_llrs(inner_decisions, inner_phis,
                base_llr: float = 2.0,
                phi_scale: float = 8.0) -> np.ndarray:
    """
    Convert inner hard decisions + soft phi to per-qubit LLRs.

    inner_decisions : list/array of 0/1 (c_hat)
        0 = 'no logical error' decision from inner decoder
        1 = 'logical error' decision.
    inner_phis : list/array of floats
        Soft output signal φ for each block.

    Returns
    -------
    initial_llrs : ndarray
        Prior LLR for each outer variable.
        Positive LLR = more confident 'no error', negative = 'error'.
    """
    c_hat = np.array(inner_decisions, dtype=int)
    phi_array = np.array(inner_phis, dtype=float)

    # Larger |phi| -> more confidence
    confidence_factor = base_llr + phi_array / phi_scale
    # (1 - 2*c_hat) = +1 if c_hat=0, -1 if c_hat=1
    initial_llrs = (1 - 2 * c_hat) * confidence_factor
    return initial_llrs


def bp_decode_outer_soft(code: LPCode,
                         inner_decisions,
                         inner_phis,
                         syndrome,
                         max_iterations: int = 50
                        ):
    Hz = get_Hz_matrix(code)
    prior_llrs = phi_to_llrs(inner_decisions, inner_phis)
    return bp_decode_with_priors(Hz, prior_llrs, syndrome, max_iterations=max_iterations)


def sample_inner_decisions_and_phis(
        p_physical: float,
        decoder_name: str,
        d_inner: int,
        outer_n: int,
        N: int = 1
):
    inner_decisions = []
    inner_phis = []
    for _ in range(outer_n):
        result_dict = correlation.get_phi_inner(p_physical, decoder_name, d_inner, N=N)
        c_hat = 1 - result_dict['no_error']  # 0 = no logical error, 1 = logical error
        phi = result_dict['phi']
        inner_decisions.append(c_hat)
        inner_phis.append(phi)
    return np.array(inner_decisions, dtype=int), np.array(inner_phis, dtype=float)


def estimate_inner_logical_rate(
        p_physical: float,
        decoder_name: str,
        d_inner: int,
        num_samples: int = 2000,
) -> float:
    """
    Estimate p_L(p_physical) = Pr[logical error] for the inner code
    by sampling the inner decoder num_samples times.

    Uses your existing get_phi_inner(p, decoder_name, d_inner, N=1),
    which returns a dict with key 'no_error' in {0,1}.
    """
    failures = 0
    for _ in range(num_samples):
        res = correlation.get_phi_inner(p_physical, decoder_name, d_inner, N=1)
        # in your code: c_hat = 1 - res['no_error']
        logical_error = 1 - res['no_error']
        failures += logical_error
    return failures / num_samples


def simulate_outer_hard_uniform_prior(
        code: LPCode,
        p_phys_list,
        decoder_name: str,
        d_inner: int,
        inner_samples_per_p: int = 2000,
        outer_trials_per_p: int = 100,
        max_iterations: int = 50,
):
    """
    For each physical error rate p_phys:

      1. Estimate p_block = p_L(p_phys) from the inner code.
      2. Treat the outer channel as BSC(p_block).
      3. Run your existing bp_decode_outer_lp_hard to get failure rate.

    This matches the 'hard information = uniform prior from marginal logical
    error rate' description in the paper.
    """
    Hz = get_Hz_matrix(code)
    n = Hz.shape[1]
    rng = np.random.default_rng()

    fail_rates_hard = []

    for p_phys in p_phys_list:
        print(f"\n[HARD] p_phys = {p_phys:.3f}")

        # 1) estimate marginal logical error rate of the inner code
        p_block = estimate_inner_logical_rate(
            p_physical=p_phys,
            decoder_name=decoder_name,
            d_inner=d_inner,
            num_samples=inner_samples_per_p,
        )
        print(f"  estimated p_block (p_L) = {p_block:.4g}")

        # 2) simulate outer BSC(p_block) + BP
        fails = 0
        for _ in range(outer_trials_per_p):
            true_error = (rng.random(n) < p_block).astype(int)
            syndrome = (Hz @ true_error) % 2

            est_error, _ = bp_decode_outer_lp_hard(
                code,
                syndrome=syndrome,
                p_block=p_block,
                max_iterations=max_iterations,
            )

            if not np.array_equal(est_error, true_error):
                fails += 1

        fail_rates_hard.append(fails / outer_trials_per_p)

    return np.array(fail_rates_hard)


def simulate_outer_soft_information(
        code: LPCode,
        p_phys_list,
        decoder_name: str,
        d_inner: int,
        trials_per_p: int = 100,
        N_inner: int = 1,
        max_iterations: int = 50,
):
    """
    For each physical error rate p_phys:

      • For each trial:
          - For each outer qubit i, run the inner code to get (c_hat_i, phi_i).
            Here c_hat_i is the noisy 'channel output' bit for an underlying
            true outer codeword bit 0.
          - Run the soft-information outer decoder and try to recover the
            all-zero codeword.
          - Declare success iff the outer decoder outputs the all-zero vector.

      • Return the logical failure rate vs p_phys for the soft-information decoder.
    """
    Hz = get_Hz_matrix(code)
    m_outer, n_outer = Hz.shape

    fail_rates_soft = []

    for p_phys in p_phys_list:
        print(f"\n[SOFT] p_phys = {p_phys:.3f}")
        fails_soft = 0

        for _ in range(trials_per_p):
            # 1) Inner runs for each outer block: noisy observations
            inner_decisions, inner_phis = sample_inner_decisions_and_phis(
                p_physical=p_phys,
                decoder_name=decoder_name,
                d_inner=d_inner,
                outer_n=n_outer,
                N=N_inner,
            )

            # True outer codeword is 0^n, so desired syndrome is 0
            syndrome_zero = np.zeros(m_outer, dtype=int)

            # 2) Soft outer decoding using per-block (c_hat_i, phi_i)
            soft_decisions, soft_llrs = bp_decode_outer_soft(
                code=code,
                inner_decisions=inner_decisions.tolist(),
                inner_phis=inner_phis.tolist(),
                syndrome=syndrome_zero,
                max_iterations=max_iterations,
            )

            # 3) Success: did we recover the all-zero codeword?
            soft_success = np.all(soft_decisions == 0)
            if not soft_success:
                fails_soft += 1

        fail_rates_soft.append(fails_soft / trials_per_p)

    return np.array(fail_rates_soft)




if __name__ == '__main__':
    code = build_qc_lifted_product_code(lift_size = 15)
    Hz = get_Hz_matrix(code)
    n_outer = Hz.shape[1]

    p_vals_hard = np.linspace(0.055, 0.075, 8)

    fail_hard = simulate_outer_hard_uniform_prior(
        code,
        p_vals_hard,
        decoder_name="unionfind",
        d_inner=5,
        outer_trials_per_p=10
    )

    print("failure rate hard info:", str(fail_hard))

    p_vals_soft = np.linspace(0.065, 0.085, 8)

    fail_soft = simulate_outer_soft_information(
        code,
        p_phys_list=p_vals_soft,
        decoder_name="unionfind",
        d_inner=5,
        trials_per_p=10,
        N_inner=1,
    )

    print("failure rate soft info:", str(fail_soft))

    plt.figure()
    plt.loglog(p_vals_hard, fail_hard, marker='o', label="hard information")
    plt.loglog(p_vals_soft, fail_soft, marker='s', label="soft information")

    plt.xlabel("Physical error rate p ")
    plt.ylabel("Logical failure probability")
    plt.title("Union Find ")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig("UF_plot.pdf")
    plt.show()

    with open("output.txt", "w") as f:
        f.write("regular union find: \n")
        f.write("soft failure rates: ")
        f.write(" ".join(map(str, fail_soft)))
        f.write("\n")
        f.write("hard failure rates: ")
        f.write(" ".join(map(str, fail_hard)))
        f.write("\n")