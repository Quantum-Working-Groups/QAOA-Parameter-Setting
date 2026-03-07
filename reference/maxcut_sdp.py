"""
MAXCUT SDP relaxation with Goemans-Williamson random hyperplane rounding.

SDP formulation (Goemans-Williamson 1995):
    max  (1/4) tr(L X)
    s.t. X >> 0
         X_ii = 1  for all i

where L is the weighted graph Laplacian.
"""

import numpy as np
import cvxpy as cp
import time


def build_laplacian(n, edges):
    """Build weighted graph Laplacian from edge list [(u, v, w), ...]."""
    L = np.zeros((n, n))
    for u, v, w in edges:
        L[u, u] += w
        L[v, v] += w
        L[u, v] -= w
        L[v, u] -= w
    return L


def solve_maxcut_sdp(n, L):
    """
    Solve the MAXCUT SDP relaxation.

    Returns:
        X_val   : (n, n) ndarray, the SDP solution matrix (or None on failure)
        sdp_val : float, optimal SDP objective value (upper bound on MAXCUT)
        solve_t : float, wall-clock solve time in seconds
        status  : str, solver status string
    """
    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> 0, cp.diag(X) == np.ones(n)]
    prob = cp.Problem(cp.Maximize(0.25 * cp.trace(L @ X)), constraints)

    t0 = time.perf_counter()
    # Prefer CLARABEL (faster) if available, fall back to SCS.
    preferred = [s for s in ("CLARABEL", "SCS") if s in cp.installed_solvers()]
    solver_kwargs = {"SCS": {"eps": 1e-5}, "CLARABEL": {}}.get(preferred[0], {})
    prob.solve(solver=getattr(cp, preferred[0]), **solver_kwargs)
    if prob.status in ("infeasible", "unbounded") or X.value is None:
        # Last resort: try every available SDP-capable solver.
        for solver_name in cp.installed_solvers():
            if solver_name in preferred:
                continue
            try:
                prob.solve(solver=getattr(cp, solver_name, None))
                if X.value is not None:
                    break
            except Exception:
                continue
    solve_t = time.perf_counter() - t0

    return X.value, prob.value, solve_t, prob.status


def factorize_sdp_solution(X, eps=1e-8):
    """
    Compute embedding V (n x n) such that V @ V.T ≈ X.
    V[i, :] is the embedding vector for node i.

    Uses eigendecomposition for numerical robustness.
    """
    X_sym = (X + X.T) / 2
    eigvals, eigvecs = np.linalg.eigh(X_sym)
    eigvals = np.maximum(eigvals, 0)
    return eigvecs * np.sqrt(eigvals)


def run_roundings(V, edges, checkpoints=(10, 100, 1000, 10000), seed=0, batch_size=256):
    """
    Run random hyperplane roundings cumulatively up to max(checkpoints).

    At each rounding, a random hyperplane normal r ~ N(0, I) is sampled and
    each node i is assigned sign(V[i, :] . r).  The best cut found so far is
    recorded at every checkpoint.

    Args:
        V           : (n, d) embedding from factorize_sdp_solution
        edges       : list of (u, v, w) tuples
        checkpoints : rounding counts at which to record results
        seed        : random seed
        batch_size  : roundings processed per vectorised batch

    Returns:
        dict mapping checkpoint (int) ->
            {best_cut_value, best_cut (list, ±1), rounding_time_seconds}
    """
    rng = np.random.default_rng(seed)
    n, d = V.shape
    total = max(checkpoints)
    checkpoint_set = set(checkpoints)

    # Pre-build edge arrays for vectorised cut computation.
    if edges:
        eu = np.array([e[0] for e in edges], dtype=np.intp)
        ev = np.array([e[1] for e in edges], dtype=np.intp)
        ew = np.array([e[2] for e in edges], dtype=float)
    else:
        eu = ev = ew = np.array([], dtype=float)

    best_value = -np.inf
    best_assignment = None
    results = {}

    t0 = time.perf_counter()
    done = 0

    while done < total:
        batch = min(batch_size, total - done)

        # Sample random hyperplane normals: (d, batch)
        R = rng.standard_normal((d, batch))
        # Node projections: (n, batch)
        scores = V @ R
        # Assignments in {+1, -1}: (n, batch)
        asgn = np.sign(scores)
        asgn[asgn == 0] = 1.0

        # Cut values for each rounding in the batch: (batch,)
        if len(eu) > 0:
            prod = asgn[eu, :] * asgn[ev, :]          # (n_edges, batch)
            cut_vals = (ew @ (1.0 - prod)) / 2.0       # (batch,)
        else:
            cut_vals = np.zeros(batch)

        for j in range(batch):
            done += 1
            if cut_vals[j] > best_value:
                best_value = float(cut_vals[j])
                best_assignment = asgn[:, j].copy()

            if done in checkpoint_set:
                results[done] = {
                    "best_cut_value": best_value,
                    "best_cut": best_assignment.astype(int).tolist(),
                    "rounding_time_seconds": time.perf_counter() - t0,
                }

    return results
