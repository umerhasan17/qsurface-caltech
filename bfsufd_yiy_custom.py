import numpy as np
from typing import Dict, List, Tuple, Any, Sequence
import matplotlib.pyplot as plt
import correlation
import heapq


def build_rotated_surface_code_Z(d: int) -> np.ndarray:
    """
    Build Z-check matrix H_Z for a rotated planar surface code of distance d.

    Data qubits live on vertices of a d x d grid (n = d^2).
    Z stabilizers live on faces (i, j) with 0 <= i, j < d-1 and (i + j) even,
    and act on the 4 vertices of the face.

    Returns
    -------
    H_Z : ndarray of shape (m, n)
        Binary parity-check matrix.
    """
    n = d * d
    rows = []

    def q(i: int, j: int) -> int:
        return i * d + j

    for i in range(d - 1):
        for j in range(d - 1):
            if (i + j) % 2 == 0:
                row = np.zeros(n, dtype=np.uint8)
                row[q(i, j)]     = 1
                row[q(i+1, j)]   = 1
                row[q(i, j+1)]   = 1
                row[q(i+1, j+1)] = 1
                rows.append(row)

    H_Z = np.array(rows, dtype=np.uint8)
    return H_Z


def build_logical_Z_vector(d: int) -> np.ndarray:
    """
    Logical Z operator for rotated planar code: Z on first column of vertices.

    Returns
    -------
    LZ : ndarray of shape (n,)
        Binary vector indicating which qubits logical Z acts on.
    """
    n = d * d
    LZ = np.zeros(n, dtype=np.uint8)
    for i in range(d):
        LZ[i * d + 0] = 1  # first column
    return LZ


class UnionFind:
    """
    Disjoint-set (Union-Find) with size + parity (for syndrome parity).
    Nodes: we'll have m check nodes and n qubit nodes.
    """

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=int)
        self.size   = np.ones(n, dtype=int)
        self.parity = np.zeros(n, dtype=int)  # parity[root] = (# odd checks in cluster) mod 2

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        """
        Weighted union by size; parity is XORed.
        Returns the root of the merged cluster.
        """
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        # ra is new root
        self.parent[rb] = ra
        self.size[ra]  += self.size[rb]
        self.parity[ra] ^= self.parity[rb]
        return ra

def bfs_unionfind_decode(H: np.ndarray, syndrome: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    BFS–Union–Find decoder on the Tanner graph of H.

    Graph:
      - Nodes 0..m-1     : check nodes
      - Nodes m..m+n-1   : qubit nodes

    We:
      1. Start with clusters on odd-syndrome checks.
      2. Grow clusters via BFS, merging clusters when they touch.
      3. Stop growing "valid" (even-parity) clusters (Alg. 2-like).
      4. For each connected component of the visited Tanner graph,
         solve H_sub x = syndrome_sub over F2 by Gaussian elimination.
      5. Define a soft metric φ as the size of the largest qubit cluster
         (number of qubits in the largest component), then rescaled to ~[0, 20].

    This φ is a monotone proxy for the "cluster size / path cost" in the paper.

    Parameters
    ----------
    H : (m, n) ndarray over F2
        Parity-check matrix for the inner code.
    syndrome : (m,) ndarray over F2
        Measured stabilizer outcomes.

    Returns
    -------
    e_hat : (n,) ndarray over F2
        Estimated error pattern.
    info : dict
        Contains:
            - 'components': dict[root] -> {'checks': [...], 'qubits': [...]}
            - 'max_cluster_qubits': int
            - 'phi': float (rescaled soft metric)
    """
    H = (H % 2).astype(np.uint8)
    syndrome = (syndrome % 2).astype(np.uint8)
    m, n = H.shape

    num_nodes = m + n  # check nodes + qubit nodes

    # Build Tanner graph adjacency
    neighbors: List[List[int]] = [[] for _ in range(num_nodes)]
    for chk in range(m):
        qs = np.nonzero(H[chk])[0]
        for q in qs:
            vq = m + q
            neighbors[chk].append(vq)
            neighbors[vq].append(chk)

    uf = UnionFind(num_nodes)
    visited = np.zeros(num_nodes, dtype=bool)
    queue: List[int] = []

    # Initialize clusters on odd syndrome checks
    odd_checks = np.nonzero(syndrome)[0]
    for v in odd_checks:
        visited[v] = True
        queue.append(v)
        uf.parity[v] = 1  # cluster containing this check is odd

    def has_invalid_cluster() -> bool:
        """Is there any visited cluster with odd parity?"""
        seen = set()
        for u in range(num_nodes):
            if not visited[u]:
                continue
            r = uf.find(u)
            if r in seen:
                continue
            seen.add(r)
            if uf.parity[r] == 1:
                return True
        return False

    # BFS growth loop (Alg. 2 style: skip valid clusters)
    i = 0
    while i < len(queue) and has_invalid_cluster():
        u = queue[i]
        i += 1

        ru = uf.find(u)
        if uf.parity[ru] == 0:
            # cluster already valid; do not grow further
            continue

        for v in neighbors[u]:
            rv = uf.find(v)
            if rv != ru:
                # merge clusters and propagate parity
                ru = uf.union(ru, rv)
            if not visited[v]:
                visited[v] = True
                queue.append(v)

    # Build components from visited nodes
    components: Dict[int, Dict[str, List[int]]] = {}
    for node in range(num_nodes):
        if not visited[node]:
            continue
        r = uf.find(node)
        if r not in components:
            components[r] = {"checks": [], "qubits": []}
        if node < m:
            components[r]["checks"].append(node)
        else:
            components[r]["qubits"].append(node - m)

    # Solve each component H_sub x = s_sub over F2
    e_hat = np.zeros(n, dtype=np.uint8)

    for comp in components.values():
        checks = comp["checks"]
        qubits = comp["qubits"]

        if not checks or not qubits:
            continue

        H_sub = H[np.ix_(checks, qubits)]
        s_sub = syndrome[checks]

        rows, cols = H_sub.shape
        # augmented matrix [H_sub | s_sub]
        A = np.concatenate([H_sub, s_sub.reshape(-1, 1)], axis=1).astype(np.uint8)

        r = c = 0
        pivots: List[Tuple[int, int]] = []

        while r < rows and c < cols:
            pivot = None
            for rr in range(r, rows):
                if A[rr, c] == 1:
                    pivot = rr
                    break
            if pivot is None:
                c += 1
                continue

            if pivot != r:
                A[[r, pivot]] = A[[pivot, r]]

            pivots.append((r, c))

            for rr in range(rows):
                if rr != r and A[rr, c] == 1:
                    A[rr, :] ^= A[r, :]

            r += 1
            c += 1

        # particular solution: set all free vars = 0
        x = np.zeros(cols, dtype=np.uint8)
        for rr, cc in pivots:
            x[cc] = A[rr, -1]

        e_hat[qubits] ^= x

    # Soft metric: largest qubit cluster size, rescaled to ≈ [0, 20]
    max_cluster_qubits = max((len(c["qubits"]) for c in components.values()), default=0)
    # For a distance-d code, n = d^2; scale linearly so n ↦ 20
    # (you can tweak this scale, but it matches your old 0..20 binning)
    n_total = n
    scale = 20.0 / max(1, n_total)
    phi = 20- max_cluster_qubits * scale

    info = {
        "components": components,
        "max_cluster_qubits": max_cluster_qubits,
        "phi": float(phi),
    }
    return e_hat, info

def bfsufd_get_phi_inner(p_bitflip: float, d: int = 5, N: int = 1) -> Dict[str, float]:
    """
    BFS–Union–Find inner decoder wrapper, compatible with your original
    get_phi_inner API.

    For each of N repetitions:
      - draw iid Z-errors with probability p_bitflip on each of the d^2 qubits
      - compute syndrome for Z-check matrix H_Z
      - run BFS–Union–Find decoder on the Tanner graph
      - compute residual error and logical Z error
      - collect 'no_error' (1/0) and φ from BFS-UFD clusters

    If N > 1, returns averaged 'no_error' and 'phi' (just like your existing code).

    Returns
    -------
    result : dict
        {
            "no_error": float in [0,1],
            "phi": float
        }
    """
    H_Z = build_rotated_surface_code_Z(d)
    logical_Z = build_logical_Z_vector(d)
    m, n = H_Z.shape

    rng = np.random.default_rng()

    no_error_sum = 0.0
    phi_sum = 0.0

    for _ in range(N):
        # sample physical bit-flips (Z-errors in CSS picture)
        e_true = (rng.random(n) < p_bitflip).astype(np.uint8)

        # syndrome
        syndrome = (H_Z @ e_true) % 2

        # BFS-UFD decode on Tanner graph
        e_hat, info = bfs_unionfind_decode(H_Z, syndrome)

        # residual error
        residual = (e_true ^ e_hat) % 2

        # logical Z error? (1 = error, 0 = no error)
        logical_error = int(np.dot(logical_Z, residual) % 2)
        no_error = 1 - logical_error  # match your convention

        phi = info["phi"]

        no_error_sum += no_error
        phi_sum += phi

    return {
        "no_error": no_error_sum / N,
        "phi": phi_sum / N,
    }



if __name__ == "__main__":
    p_bitflip = 0.05
    max_iterations = 10000
    inner_hard_decisions = []
    inner_phi_scores = []

    for _ in range(max_iterations):
        result = bfsufd_get_phi_inner(p_bitflip, d=5, N=1)
        c_hat = 1 - result["no_error"]  # 1 = logical error
        phi   = result["phi"]

        inner_hard_decisions.append(c_hat)
        inner_phi_scores.append(phi)

    phis = np.array(inner_phi_scores, dtype=float)
    Ls   = np.array(inner_hard_decisions, dtype=int)

    avg_phi, log_like, counts = correlation.estimate_pL_vs_phi(
        phis=phis,
        Ls=Ls,
        bin_edges=np.linspace(0, 20, 20),
    )

    plt.figure()
    plt.plot(avg_phi, log_like, marker="o", linestyle="none")
    plt.xlabel("soft output φ (binned)")
    plt.ylabel("log logical error rate")
    plt.title(f"BFS-UFD (d={d}), p_bitflip={p_bitflip}")
    plt.grid(True)
    plt.show()
