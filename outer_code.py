import numpy as np

from correlation import get_phi_inner


def circulant_shift_matrix(L):
    """Permutation matrix W with Wij = 1 if i+1 ≡ j (mod L)."""
    W = np.zeros((L, L), dtype=np.uint8)
    for i in range(L):
        W[i, (i + 1) % L] = 1
    return W


def poly_to_circulant(exp, L, W):
    """
    Map x^exp in F2[x]/(x^L-1) to the LxL circulant matrix W^exp.
    exp can be an integer or list of ints (XORed).
    """
    if isinstance(exp, int):
        exps = [exp]
    else:
        exps = list(exp)

    M = np.zeros((L, L), dtype=np.uint8)
    for e in exps:
        M ^= np.linalg.matrix_power(W, e % L).astype(np.uint8)
    return M


def build_qclp_hz_hx(lift_size=31):
    """
    Construct (HZ, HX) for the QCLP code used in the paper:
    - lift size ℓ = 31
    - base matrix A given in Eq. (C3).

    Returns
    -------
    HZ, HX : np.ndarray
        Binary parity-check matrices for the CSS code.
    """
    L = lift_size
    W = circulant_shift_matrix(L)

    # Base matrix A over F2[x]/(x^L-1), written as exponents of x
    # Eq. (C3) in the paper
    A_exps = [
        [1, 2, 4, 8, 16],
        [5, 10, 20, 9, 18],
        [25, 19, 7, 14, 28],
    ]
    A_exps = np.array(A_exps, dtype=int)
    m, n = A_exps.shape  # 3 x 5

    # Build the block matrix ρ(A): each entry A_ij -> LxL circulant
    # We'll store A as blocks and then form ρ(A⊗I) and ρ(I⊗A*)
    blocks_A = [[poly_to_circulant(A_exps[i, j], L, W)
                 for j in range(n)]
                for i in range(m)]

    # Conjugate transpose A* (negating exponents in the group algebra).
    # Over F2[x]/(x^L-1), this corresponds to mapping x^e -> x^{-e} = x^(L-e).
    A_star_exps = (-A_exps) % L
    blocks_A_star = [[poly_to_circulant(A_star_exps[i, j], L, W)
                      for j in range(n)]
                     for i in range(m)]

    # Now build HZ and HX as in Eq. (C2):
    #  HZ = [ ρ(A ⊗ I)   ρ(I ⊗ A*) ]
    #  HX^T = [ ρ(I ⊗ A*) ; ρ(A ⊗ I) ]
    #
    # A ⊗ I and I ⊗ A* expand A and A* into block matrices of size (mL x nL).

    # ρ(A ⊗ I): block (i,j) = blocks_A[i][j]
    top_left = np.block(blocks_A)

    # ρ(I ⊗ A*): block diagonal with A* blocks repeated along diagonal
    # Here I is size m (or n) over the group algebra; for the lifted product
    # construction, the shapes work out so that:
    #   HZ has shape (m*n*L, (m+n)*L)  -> total n_phys = 1054
    #
    # To keep the implementation simple and match [19], we use the standard
    # lifted product construction via Kronecker products of A and A*.
    #
    # For brevity, we directly use the formulas via kron.
    A_bin = np.block(blocks_A)  # (mL x nL)
    A_star_bin = np.block(blocks_A_star)

    # A ⊗ I and I ⊗ A* over F2 (Kronecker products)
    A_kron_I = np.kron(A_bin, np.eye(L, dtype=np.uint8))
    I_kron_Astar = np.kron(np.eye(A_star_bin.shape[0], dtype=np.uint8), A_star_bin)

    # Same shapes for the dual block
    I_kron_A = np.kron(np.eye(A_bin.shape[0], dtype=np.uint8), A_bin)
    Astar_kron_I = np.kron(A_star_bin, np.eye(L, dtype=np.uint8))

    # Build HZ and HX from these blocks (mod 2)
    HZ = np.concatenate([A_kron_I, I_kron_Astar], axis=1) % 2
    HX = np.concatenate([I_kron_A, Astar_kron_I], axis=1) % 2

    return HZ.astype(np.uint8), HX.astype(np.uint8)


def bp_decode_binary(H, syndrome, prior_p, max_iters=100, tol=1e-6):
    """
    Sum-product BP decoder on a binary LDPC code defined by parity-check H.

    Parameters
    ----------
    H : np.ndarray of shape (m, n), dtype uint8
        Parity-check matrix (we use HZ for bit-flip errors on a CSS code).
    syndrome : array-like length m, bits {0,1}
        Measured syndrome.
    prior_p : float or array-like length n
        Prior flip probabilities for each variable node:
            p_i = Pr(bit i = 1)  (i.e. 'error' on that qubit).
        If float, the same p is used for all qubits (hard info).
    max_iters : int
        Maximum number of BP iterations.
    tol : float
        Stopping tolerance for change in beliefs (not strictly necessary).

    Returns
    -------
    est_error : np.ndarray length n, dtype uint8
        Estimated error pattern (bits).
    success : bool
        True if H @ est_error % 2 == syndrome at termination.
    """
    H = np.asarray(H, dtype=np.uint8)
    s = np.asarray(syndrome, dtype=np.uint8)
    m, n = H.shape

    # Broadcast prior_p to per-variable array
    if np.isscalar(prior_p):
        p = np.full(n, float(prior_p))
    else:
        p = np.asarray(prior_p, dtype=float)
        assert p.shape == (n,)

    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)

    # Prior LLRs γ_i = ln( (1 - p_i) / p_i )
    gamma = np.log((1.0 - p) / p)

    # Adjacency lists
    H_bool = H.astype(bool)
    var_neighbors = [np.where(H_bool[:, i])[0] for i in range(n)]  # checks touching each var
    check_neighbors = [np.where(H_bool[j, :])[0] for j in range(m)]  # vars touching each check

    # Messages m_{v->c} and m_{c->v} stored in dicts keyed by (var, check)
    m_vc = {}
    m_cv = {}

    # Initialize messages v->c with prior γ_i, and c->v with 0
    for i in range(n):
        for j in var_neighbors[i]:
            m_vc[(i, j)] = gamma[i]
            m_cv[(j, i)] = 0.0

    est = np.zeros(n, dtype=np.uint8)

    for it in range(max_iters):
        # --- Check node updates: m_{c->v} ---
        for j in range(m):
            sj = int(s[j])
            neigh = check_neighbors[j]
            if len(neigh) == 0:
                continue

            # Precompute tanh( m_{v'->c} / 2 ) for all neighbours
            tanhs = {i: np.tanh(m_vc[(i, j)] / 2.0) for i in neigh}

            for i in neigh:
                # Product over all neighbors except i
                prod = 1.0
                for i2 in neigh:
                    if i2 == i:
                        continue
                    prod *= tanhs[i2]

                # Clip prod into (-1,1) to avoid numerical issues
                prod = np.clip(prod, -0.999999, 0.999999)

                # m_{c->v} = (−1)^s_j * 2 * atanh(prod)
                msg = ((-1) ** sj) * np.arctanh(prod) * 2.0
                m_cv[(j, i)] = msg

        # --- Variable node updates: m_{v->c} and marginal LLRs ---
        llr_total = np.zeros(n)

        for i in range(n):
            neigh_c = var_neighbors[i]

            # Total LLR γ_i + sum_{c} m_{c->v}
            total = gamma[i]
            for j in neigh_c:
                total += m_cv[(j, i)]
            llr_total[i] = total

            # Messages to checks: exclude recipient check
            for j in neigh_c:
                msg = gamma[i]
                for j2 in neigh_c:
                    if j2 == j:
                        continue
                    msg += m_cv[(j2, i)]
                m_vc[(i, j)] = msg

        # --- Hard decision from marginals ---
        new_est = (llr_total < 0).astype(np.uint8)

        # Check if syndrome is satisfied
        if np.all((H @ new_est) % 2 == s):
            est = new_est
            return est, True

        # Optional convergence check (change in beliefs)
        if np.max(np.abs(llr_total - gamma)) < tol:
            est = new_est
            break

        est = new_est

    # Final check
    success = np.all((H @ est) % 2 == s)
    return est, bool(success)


def make_phi_to_p_func(phi_grid, p_grid):
    """
    Given calibration arrays (phi_grid, p_grid), return a function
    phi -> p(phi) via 1D interpolation.

    phi_grid : array of representative phi values (e.g. avg_phi_per_bin)
    p_grid   : array of corresponding logical error probabilities
               estimated from inner-code simulations.
    """
    phi_grid = np.asarray(phi_grid, dtype=float)
    p_grid = np.asarray(p_grid, dtype=float)

    def phi_to_p(phi_array):
        phi_array = np.asarray(phi_array, dtype=float)
        # interpolate in phi; clamp to endpoints outside range
        return np.interp(phi_array, phi_grid, p_grid,
                         left=p_grid[0], right=p_grid[-1])

    return phi_to_p


def outer_decode_soft(phi_block,
                      HZ,
                      syndrome,
                      phi_grid,
                      pL_grid,
                      max_iters=100):
    """
    Decode the QCLP outer code using soft information from the inner code.

    Parameters
    ----------
    phi_block : array-like length n_phys
        Soft outputs ϕ_i from the inner surface codes (one per outer-code qubit).
    HZ : np.ndarray (m, n_phys)
        Z-check matrix of the QCLP code (bit-flip sector).
    syndrome : array-like length m
        Measured outer-code syndrome bits for the current BP problem.
    phi_grid, pL_grid : arrays
        Calibration data for mapping ϕ -> p, as returned by the inner-code
        simulations and binning.
    max_iters : int
        Max BP iterations.

    Returns
    -------
    est_error : np.ndarray length n_phys
        Estimated X-error pattern on the outer code.
    success : bool
        True if HZ @ est_error == syndrome (mod 2).
    """
    phi_block = np.asarray(phi_block, dtype=float)
    HZ = np.asarray(HZ, dtype=np.uint8)
    syndrome = np.asarray(syndrome, dtype=np.uint8)

    # Build phi -> p(ϕ) mapping and compute per-qubit priors
    phi_to_p = make_phi_to_p_func(phi_grid, pL_grid)
    prior_p = phi_to_p(phi_block)  # array length n_phys

    # Run BP with non-uniform priors
    est_error, success = bp_decode_binary(HZ, syndrome, prior_p, max_iters=max_iters)
    return est_error, success


def outer_decode_hard(e_block,
                      HZ,
                      syndrome,
                      p_marginal,
                      max_iters=100):
    """
    Decode the QCLP outer code using only a uniform marginal prior.

    Parameters
    ----------
    e_block : array-like length n_phys
        Hard inner logical error indicators (not directly used here except
        for consistency; the decoder only uses syndrome + uniform prior).
        You might still want this around for bookkeeping.
    HZ : np.ndarray (m, n_phys)
        Z-check matrix of the QCLP code.
    syndrome : array-like length m
        Measured outer-code syndrome bits.
    p_marginal : float
        Marginal inner-code logical error rate, used as a *uniform* prior.
        This is exactly what the paper calls “hard information”. :contentReference[oaicite:6]{index=6}
    max_iters : int
        Max BP iterations.

    Returns
    -------
    est_error : np.ndarray length n_phys
        Estimated X-error pattern.
    success : bool
        True if HZ @ est_error == syndrome (mod 2).
    """
    HZ = np.asarray(HZ, dtype=np.uint8)
    syndrome = np.asarray(syndrome, dtype=np.uint8)
    # Uniform prior over all variables
    est_error, success = bp_decode_binary(HZ, syndrome, prior_p=float(p_marginal),
                                          max_iters=max_iters)
    return est_error, success



import numpy as np

class InnerJointDistribution:
    """
    Stores samples (phi, e) from the inner surface code:
      - phi: soft output
      - e:   logical failure bit (0 or 1)
    and lets you resample i.i.d. from this empirical distribution.
    """
    def __init__(self, phis, errors):
        phis = np.asarray(phis, dtype=float)
        errors = np.asarray(errors, dtype=int)
        assert phis.shape == errors.shape
        self.phis = phis
        self.errors = errors
        self.N = len(phis)

    def sample(self, size):
        """Return arrays (phi_samples, error_samples) of length size,
        sampled i.i.d. from the empirical joint distribution."""
        idx = np.random.randint(0, self.N, size=size)
        return self.phis[idx], self.errors[idx]

    @property
    def marginal_p_surface(self):
        """Global inner logical error rate."""
        return self.errors.mean()


# inner_dist = InnerJointDistribution(phis_inner, Ls_inner)
# print("Global inner logical error rate p_surface =", inner_dist.marginal_p_surface)




def simulate_outer_code_failure_rate(inner_dist,
                                     k_blocks,
                                     num_trials,
                                     decoder_type="soft"):
    """
    Estimate the logical failure rate of the outer code:

      - inner_dist: InnerJointDistribution
      - k_blocks:   number of inner code blocks feeding into the outer code
      - num_trials: Monte Carlo trials
      - decoder_type: "soft" or "hard"

    Returns
    -------
    p_outer_hat : float
        Estimated logical failure rate of the outer code.
    """

    failures = 0
    p_marginal = inner_dist.marginal_p_surface  # for 'hard' prior

    for _ in range(num_trials):
        # Sample noise for all inner blocks participating in the outer code
        phi_block, e_block = inner_dist.sample(size=k_blocks)

        if decoder_type == "soft":
            # YOU implement this:
            # should use both phi_block and (optionally) e_block
            # to run BP with soft priors and decide if outer logical error
            logical_fail_outer = outer_decode_soft(phi_block, e_block)
        elif decoder_type == "hard":
            # YOU implement this:
            # should ignore phi_block; use only e_block and p_marginal
            logical_fail_outer = outer_decode_hard(e_block, p_marginal)
        else:
            raise ValueError("decoder_type must be 'soft' or 'hard'.")

        failures += int(logical_fail_outer)

    return failures / num_trials





def at_least_one_failure_curve(p_surface_vals, k_blocks):
    p_surface_vals = np.asarray(p_surface_vals, dtype=float)
    return 1.0 - (1.0 - p_surface_vals) ** k_blocks


import matplotlib.pyplot as plt
import numpy as np

def plot_figure5_style(p_surface_vals,
                       p_outer_soft_vals,
                       p_outer_hard_vals,
                       k_blocks):
    """
    Make a Fig. 5-style plot:
      - x-axis: inner logical error rate p_surface
      - curves: outer logical failure rates (soft, hard),
                and P(at least one of k inner codes fails)
    """
    p_surface_vals = np.asarray(p_surface_vals)
    p_outer_soft_vals = np.asarray(p_outer_soft_vals)
    p_outer_hard_vals = np.asarray(p_outer_hard_vals)

    # Green curve: naive 'any inner block fails' bound
    p_any_inner_fail = at_least_one_failure_curve(p_surface_vals, k_blocks)

    fig, ax = plt.subplots()

    ax.loglog(p_surface_vals, p_outer_soft_vals, marker='o', label="Soft info")
    ax.loglog(p_surface_vals, p_outer_hard_vals, marker='s', label="Hard info")
    ax.loglog(p_surface_vals, p_any_inner_fail, marker='^', label=f"At least one of k={k_blocks} fails")

    ax.set_xlabel("Logical error rate of inner surface code  $p_{\\mathrm{surface}}$")
    ax.set_ylabel("Logical failure rate of outer code")
    ax.grid(True, which="both", ls=":")
    ax.legend()

    # Optional: mark pseudothresholds as vertical dashed lines
    # (solve for p where p_outer = p_surface for each decoder)
    def pseudothreshold(p_surface, p_outer):
        # naive: find first crossing where p_outer < p_surface
        # and linearly interpolate in log-space
        for i in range(1, len(p_surface)):
            if p_outer[i] < p_surface[i] <= p_outer[i-1]:
                # log-linear interpolation
                x1, x2 = np.log10(p_surface[i-1]), np.log10(p_surface[i])
                y1, y2 = np.log10(p_outer[i-1]), np.log10(p_outer[i])
                # find t where log p_outer = log p_surface
                # i.e. y1 + t (y2-y1) = x1 + t (x2-x1)
                denom = (y2 - y1) - (x2 - x1)
                if abs(denom) < 1e-12:
                    continue
                t = (x1 - y1) / denom
                log_p_thresh = x1 + t * (x2 - x1)
                return 10 ** log_p_thresh
        return None

    pth_soft = pseudothreshold(p_surface_vals, p_outer_soft_vals)
    pth_hard = pseudothreshold(p_surface_vals, p_outer_hard_vals)

    if pth_soft is not None:
        ax.axvline(pth_soft, linestyle="--")
    if pth_hard is not None:
        ax.axvline(pth_hard, linestyle="--")

    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt

# --------- Assumed available from earlier code ---------
# estimate_pL_vs_phi(phis, Ls, bin_edges) -> (avg_phi_per_bin, pL_per_bin, counts_per_bin)
# InnerJointDistribution(phis, Ls) with .sample(size) and .marginal_p_surface
# make_phi_to_p_func(phi_grid, p_grid) -> callable phi_to_p(phi_array)
# outer_decode_soft(phi_block, e_block, H_Z_outer, syndrome_outer, pL_given_phi, max_iters=50)
#     -> returns (logical_failure:int, residual:np.ndarray)
# outer_decode_hard(e_block, H_Z_outer, syndrome_outer, p_surface_marginal, max_iters=50)
#     -> returns (logical_failure:int, residual:np.ndarray)
# is_logical_failure(residual) -> bool
# run_surface_decoder(p_surface, n_inner_trials) -> (phis, logical_errors)
# H_Z_outer : binary numpy array shape (m_outer, n_outer)
#
# If any of those are missing, implement/stub them before calling run_experiment.
# ------------------------------------------------------

def compute_phi_to_p_calibration(phis, Ls, n_bins=40):
    """
    Build calibration mapping phi -> p_L using binned empirical frequencies.
    Returns phi_grid (avg phi per non-empty bin), pL_grid (emp estimate per bin),
    and a callable phi_to_p(phi_array).
    """
    phis = np.asarray(phis, dtype=float)
    Ls = np.asarray(Ls, dtype=int)
    assert phis.shape == Ls.shape

    # choose bin edges spanning the data with a small margin
    phi_min, phi_max = np.min(phis), np.max(phis)
    margin = max(1e-3, 0.05 * (phi_max - phi_min) if phi_max > phi_min else 1.0)
    edges = np.linspace(phi_min - margin, phi_max + margin, n_bins + 1)

    avg_phi_per_bin, pL_per_bin, counts = estimate_pL_vs_phi(phis, Ls, edges)

    # If too many empty bins / degenerate result, fall back to a simple logistic fit
    if len(avg_phi_per_bin) < 5:
        # fallback: logistic fit on raw data (be robust)
        from scipy.optimize import curve_fit

        def logistic(phi, a, b):
            return 1.0 / (1.0 + np.exp(-(a * phi + b)))

        # to fit, we need aggregate points: take small number of quantiles
        quantiles = np.linspace(0.01, 0.99, 10)
        q_phi = np.quantile(phis, quantiles)
        q_p = []
        for q_low, q_high in zip(np.concatenate(([0.0], quantiles[:-1])), quantiles):
            mask = (phis >= np.quantile(phis, q_low)) & (phis <= np.quantile(phis, q_high))
            if np.sum(mask) == 0:
                q_p.append(np.mean(Ls))
            else:
                q_p.append(np.mean(Ls[mask]))
        q_phi = np.array(q_phi)
        q_p = np.array(q_p)
        try:
            popt, _ = curve_fit(logistic, q_phi, q_p, p0=(1.0, 0.0), maxfev=5000)
            def phi_to_p(phi_array):
                return 1.0 / (1.0 + np.exp(-(popt[0] * np.asarray(phi_array) + popt[1])))
            # create coarse grids to keep API consistent
            return q_phi, q_p, phi_to_p
        except Exception:
            # Last-resort: constant mapping equal to marginal
            p_marginal = np.mean(Ls)
            def phi_to_p(phi_array):
                return np.full(np.shape(phi_array), p_marginal, dtype=float)
            return np.array([np.mean(phis)]), np.array([p_marginal]), phi_to_p

    # Otherwise build interpolation mapping
    phi_to_p = make_phi_to_p_func(avg_phi_per_bin, pL_per_bin)
    return avg_phi_per_bin, pL_per_bin, phi_to_p


def run_experiment(
    p_surface_list,
    run_surface_decoder,
    H_Z_outer,
    is_logical_failure,
    n_inner_trials=20000,
    n_outer_trials=4000,
    k_blocks=None,
    n_phi_bins=40,
    bp_max_iters=50,
):
    """
    Run the hierarchical experiment sweeping over p_surface_list (inner-code physical error rates).

    Parameters
    ----------
    p_surface_list : iterable of float
        The list/array of inner surface-code physical error rates to sweep.
    run_surface_decoder : callable
        Function run_surface_decoder(p_surface, n_inner_trials) -> (phis, logical_errors)
        - phis: array-like length n_inner_trials of soft outputs (floats)
        - logical_errors: array-like length n_inner_trials of {0,1}
    H_Z_outer : np.ndarray (m_outer, n_outer)
        Outer code Z-check matrix.
    is_logical_failure : callable
        Function is_logical_failure(residual: np.ndarray) -> bool
        Decides if residual error is a logical failure for the outer code.
    n_inner_trials : int
        Number of inner-code Monte Carlo samples to draw for calibration per p_surface.
    n_outer_trials : int
        Number of outer-code Monte Carlo trials per p_surface value.
    k_blocks : int or None
        Number of inner blocks feeding the outer code (default: n_outer length).
    n_phi_bins : int
        Number of bins used to calibrate phi -> p mapping.
    bp_max_iters : int
        Max iterations to pass into outer decoders.

    Returns
    -------
    p_surface_vals, p_outer_soft_vals, p_outer_hard_vals
    Arrays of length len(p_surface_list).
    """

    H_Z_outer = np.asarray(H_Z_outer, dtype=int)
    m_outer, n_outer = H_Z_outer.shape
    if k_blocks is None:
        k_blocks = n_outer

    p_surface_vals = []
    p_outer_soft_vals = []
    p_outer_hard_vals = []

    for p_surf in p_surface_list:
        print(f"[run_experiment] p_surface = {p_surf}")

        # 1) Produce inner-code calibration samples
        phis, logical_errors = run_surface_decoder(p_surf, n_inner_trials)
        phis = np.asarray(phis, dtype=float)
        logical_errors = np.asarray(logical_errors, dtype=int)
        if len(phis) != len(logical_errors):
            raise ValueError("run_surface_decoder must return phis and logical_errors of same length")

        p_surface_marginal = float(np.mean(logical_errors))
        print(f"  -> inner marginal p_surface (empirical) = {p_surface_marginal:.4e} (from {len(phis)} samples)")

        # 2) Calibrate phi -> p_L mapping
        phi_grid, pL_grid, phi_to_p = compute_phi_to_p_calibration(phis, logical_errors, n_bins=n_phi_bins)

        # 3) Build empirical joint distribution for resampling into outer trials
        inner_dist = InnerJointDistribution(phis, logical_errors)

        # 4) Outer Monte Carlo
        fail_soft = 0
        fail_hard = 0

        for t in range(n_outer_trials):
            # sample k_blocks i.i.d. inner-block outcomes
            phi_block, e_block = inner_dist.sample(size=k_blocks)  # arrays length k_blocks

            # If outer code length n_outer != k_blocks, tile or truncate as needed.
            if k_blocks != n_outer:
                # simple policy: if k_blocks < n_outer, repeat samples; if >, truncate
                if k_blocks < n_outer:
                    reps = int(np.ceil(n_outer / k_blocks))
                    phi_block = np.tile(phi_block, reps)[:n_outer]
                    e_block = np.tile(e_block, reps)[:n_outer]
                else:
                    phi_block = phi_block[:n_outer]
                    e_block = e_block[:n_outer]
            # ensure arrays length match outer code
            phi_block = np.asarray(phi_block, dtype=float)
            e_block = np.asarray(e_block, dtype=int)

            # compute outer syndrome from the true outer error pattern (bit-flips)
            syndrome_outer = (H_Z_outer @ e_block) % 2

            # SOFT decoding trial
            logical_failure_soft, residual_soft = outer_decode_soft(
                phi_block,
                e_block,
                H_Z_outer,
                syndrome_outer,
                phi_to_p,           # mapping phi -> pL
                max_iters=bp_max_iters
            )
            # In case the wrapper returns (est_error, success) instead of (logical_failure, residual),
            # try to detect and adapt. But here we expect (logical_failure:int, residual:np.ndarray).
            if isinstance(logical_failure_soft, (list, tuple)) and len(logical_failure_soft) == 2:
                # previously some wrappers returned (est_error, success). Try to interpret:
                est, success_flag = logical_failure_soft
                # compute residual and logical flag consistently
                residual_soft = (e_block ^ est).astype(int)
                logical_failure_soft = int(is_logical_failure(residual_soft))

            fail_soft += int(logical_failure_soft)

            # HARD decoding trial
            logical_failure_hard, residual_hard = outer_decode_hard(
                e_block,
                H_Z_outer,
                syndrome_outer,
                p_surface_marginal,
                max_iters=bp_max_iters
            )
            if isinstance(logical_failure_hard, (list, tuple)) and len(logical_failure_hard) == 2:
                est, success_flag = logical_failure_hard
                residual_hard = (e_block ^ est).astype(int)
                logical_failure_hard = int(is_logical_failure(residual_hard))

            fail_hard += int(logical_failure_hard)

        p_outer_soft = fail_soft / float(n_outer_trials)
        p_outer_hard = fail_hard / float(n_outer_trials)

        p_surface_vals.append(p_surface_marginal)
        p_outer_soft_vals.append(p_outer_soft)
        p_outer_hard_vals.append(p_outer_hard)

        print(f"  -> outer (soft) = {p_outer_soft:.4e}, outer (hard) = {p_outer_hard:.4e}")

    return np.array(p_surface_vals), np.array(p_outer_soft_vals), np.array(p_outer_hard_vals)


# Simple Fig.5-style plotting helper (re-using earlier style)
def plot_figure5_style(p_surface_vals, p_outer_soft_vals, p_outer_hard_vals, k_blocks):
    p_any_inner_fail = 1.0 - (1.0 - p_surface_vals) ** k_blocks

    fig, ax = plt.subplots(figsize=(7,5))
    ax.loglog(p_surface_vals, p_outer_soft_vals, marker='o', label="Outer (soft info)")
    ax.loglog(p_surface_vals, p_outer_hard_vals, marker='s', label="Outer (hard info)")
    ax.loglog(p_surface_vals, p_any_inner_fail, marker='^', label=f"At least one of k={k_blocks} fails")

    ax.set_xlabel("Logical error rate of inner surface code  $p_{\\mathrm{surface}}$")
    ax.set_ylabel("Logical failure rate of outer code")
    ax.grid(True, which="both", ls=":")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ---------------- Example driver usage ----------------
if __name__ == "__main__":
    # TODO: replace the following stubs with your actual functions / matrices

    # Example stub for run_surface_decoder (replace with your simulator)
    def run_surface_decoder_stub(p_surface, n_trials):
        # toy model: phi ~ Normal(mu, sigma), and P(logical error | phi) = logistic(-phi)
        np.random.seed(int(1000 * p_surface) % 2**31)
        phis = np.random.normal(loc=0.0, scale=2.0, size=n_trials)
        probs = 1.0 / (1.0 + np.exp(phis))   # toy relation
        logical_errors = (np.random.rand(n_trials) < probs).astype(int)
        return phis, logical_errors

    # TODO: set H_Z_outer to your actual QCLP Z-check matrix (m_outer x n_outer)
    # For example placeholder:
    n_outer = 140  # small toy outer length for quick tests; replace with 1054
    m_outer = 70
    # Random small code placeholder (DO NOT use for real results)
    rng = np.random.default_rng(12345)
    H_Z_outer_stub = (rng.random((m_outer, n_outer)) < 0.1).astype(int)

    # TODO: concrete is_logical_failure implementation for your QCLP
    def is_logical_failure_stub(residual):
        # toy: declare failure if weight(residual) > threshold
        return np.sum(residual) > 8

    # Replace run_surface_decoder_stub with your real run_surface_decoder
    p_surf_list = np.array([1e-3, 2e-3, 5e-3, 1e-2])  # example sweep; replace with target points

    p_surface_vals, p_outer_soft_vals, p_outer_hard_vals = run_experiment(
        p_surf_list,
        run_surface_decoder=run_surface_decoder_stub,
        H_Z_outer=H_Z_outer_stub,
        is_logical_failure=is_logical_failure_stub,
        n_inner_trials=5000,
        n_outer_trials=2000,
        k_blocks=n_outer,      # set k_blocks to outer code length or desired value
        n_phi_bins=30,
        bp_max_iters=50,
    )

    plot_figure5_style(p_surface_vals, p_outer_soft_vals, p_outer_hard_vals, k_blocks=n_outer)







def run_experiment(
    p_surface_list,
    n_inner_trials=10000,
    n_outer_trials=2000,
    outer_block_length=100
):
    """
    Runs the experiment for the outer code with & without soft information,
    sweeping over a list of surface-code physical error rates.
    """

    p_surface_vals = []
    p_outer_soft_vals = []
    p_outer_hard_vals = []

    for p_surf in p_surface_list:
        print(f"Running for p_surface = {p_surf}")

        # --------------------------------------------------------
        # 1. Run inner decoder many times to get soft outputs φ
        # --------------------------------------------------------

        # 2. DEFINE INNER CODE PARAMETERS
        D_INNER = 5
        N_INNER = 1
        P_PHYSICAL = 0.1
        DECODER_NAME = 'unionfind'
        MAX_ITERATIONS = n_inner_trials

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

        phis = np.array(inner_phi_scores)
        logical_errors = np.array(inner_hard_decisions)
        # logical_errors are 0/1 outcomes

        # Estimate the empirical logical error rate of surface code
        p_L_surface = np.mean(logical_errors)

        # --------------------------------------------------------
        # 2. Run the outer decoder n_outer_trials times
        # --------------------------------------------------------
        outer_soft_fails = 0
        outer_hard_fails = 0

        for _ in range(n_outer_trials):
            # Draw outer_block_length inner samples
            idx = np.random.randint(0, len(phis), size=outer_block_length)
            phi_vec = phis[idx]
            hard_vec = (phis[idx] < 0).astype(int)  # example of hard decision

            # Decode with soft info
            out_soft = outer_decode_soft(phi_vec)
            if out_soft != 0:     # assume 0 = correct logical
                outer_soft_fails += 1

            # Decode with hard info
            out_hard = outer_decode_hard(hard_vec)
            if out_hard != 0:
                outer_hard_fails += 1

        p_outer_soft = outer_soft_fails / n_outer_trials
        p_outer_hard = outer_hard_fails / n_outer_trials

        # Store
        p_surface_vals.append(p_surf)
        p_outer_soft_vals.append(p_outer_soft)
        p_outer_hard_vals.append(p_outer_hard)

    return (
        np.array(p_surface_vals),
        np.array(p_outer_soft_vals),
        np.array(p_outer_hard_vals)
    )


# ------------------------------------------------------------
# PLOT GENERATOR
# ------------------------------------------------------------
def plot_results(p_surface_vals, p_outer_soft_vals, p_outer_hard_vals):
    plt.figure(figsize=(7,5))

    plt.loglog(p_surface_vals, p_outer_soft_vals, 'o-', label='Outer (soft info)')
    plt.loglog(p_surface_vals, p_outer_hard_vals, 's-', label='Outer (hard info)')

    plt.xlabel("Surface-code physical error rate  p_surface")
    plt.ylabel("Outer logical error rate")
    plt.grid(True, which='both')
    plt.legend()
    plt.title("Outer decoding performance with vs without soft information")

    plt.show()


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    # Example sweep (adjust to match the paper)
    p_surf_list = np.logspace(-4, -2.5, 6)

    p_surface_vals, p_outer_soft_vals, p_outer_hard_vals = run_experiment(
        p_surf_list,
        n_inner_trials=10000,
        n_outer_trials=2000,
        outer_block_length=200,   # tune to match outer code length
    )

    plot_results(p_surface_vals, p_outer_soft_vals, p_outer_hard_vals)


#
# # Arrays of inner logical error rates (x-axis)
# p_surface_vals = []          # e.g. len M
# # Corresponding outer logical failure rates
# p_outer_soft_vals = []
# p_outer_hard_vals = []
#
#
# k_blocks = 140  # or whatever k is for your outer code instance
#
# plot_figure5_style(p_surface_vals,
#                    p_outer_soft_vals,
#                    p_outer_hard_vals,
#                    k_blocks)
