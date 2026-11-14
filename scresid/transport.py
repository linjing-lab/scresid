"""Innovation II - CAOT: Content-guided Attentive Optimal Transport.

The original prediction path had three defects:

1. ``torch.max(G, 0)[1]`` collapsed the soft OT coupling into a hard argmax,
   discarding all column information;
2. ``cosine_similarity(...)[0]`` used only the first target cell, so every
   target cell shared one global ``delta_pred`` (cell-level delta variance = 0);
3. source controls used a uniform marginal, so sources unrelated to the target
   cell type contributed with equal weight.

CAOT replaces that with four steps:

(1) build a target-aware source marginal ``b`` anchored at the target content
    centroid;
(2) run entropic soft OT in the content subspace, and use the full column
    coupling to obtain a soft matched stimulated response per source control,
    hence ``delta_src``;
(3) give every target cell its own top-k softmax attention ``W`` over sources,
    yielding ``delta_pred_i = sum_j W_ij * delta_src_j``;
(4) gate the delta by attention confidence to restrain OOD target cells.

Only NumPy is required. POT is an optional accelerator: the module detects it
and otherwise falls back to a mathematically equivalent built-in Sinkhorn. No
package is downloaded or installed here.
"""

from __future__ import annotations

import numpy as np

try:  # optional accelerator, never installed by this package
    import ot as _pot
except ImportError:  # pragma: no cover
    _pot = None


def cosine_similarity(A, B):
    """Row-wise cosine similarity, returning ``(n_a, n_b)``."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return A @ B.T


def sinkhorn_coupling(a, b, cost, reg=0.05, num_iter=2000, tol=1e-7, use_pot=True):
    """Entropy-regularized OT coupling ``G`` with shape ``(len(a), len(b))``.

    ``cost`` should be normalized into ``[0, 1]``; smaller ``reg`` approaches
    the unregularized EMD.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    C = np.asarray(cost, dtype=np.float64)
    if C.shape != (a.shape[0], b.shape[0]):
        raise ValueError(f"cost shape {C.shape} does not match marginals ({len(a)}, {len(b)})")
    reg = max(float(reg), 1e-8)

    if use_pot and _pot is not None:
        G = _pot.sinkhorn(a, b, C, reg=reg, numItermax=num_iter, stopThr=tol)
        return np.asarray(G, dtype=np.float64)

    # Pure NumPy Sinkhorn scaling (mathematically equivalent to POT.sinkhorn).
    K = np.exp(-np.maximum(C, 0.0) / reg)
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(int(num_iter)):
        u_prev = u.copy()
        u = a / np.maximum(K @ v, 1e-300)
        v = b / np.maximum(K.T @ u, 1e-300)
        try:
            if np.max(np.abs(u - u_prev)) < tol:
                break
        except FloatingPointError:
            break
    return u[:, None] * K * v[None, :]


def caot_transfer(
    zc_target,   # (n_test, dc) content of target-type controls
    zc_source,   # (n_src,  dc) content of other-type controls
    zr_source,   # (n_src,  dr) response of other-type controls (~0 after DC-RL)
    zr_stim,     # (n_stim, dr) response of stimulated cells
    zc_stim,     # (n_stim, dc) content of stimulated cells, used for OT cost
    ratio=0.05,
    ot_reg=0.05,
    source_tau=0.25,
    attn_tau=0.10,
    confidence_floor=0.5,
    use_pot=True,
):
    """Run CAOT and return a cell-level response delta plus diagnostics."""
    zc_t = np.asarray(zc_target, dtype=np.float64)
    zc_s = np.asarray(zc_source, dtype=np.float64)
    zr_s = np.asarray(zr_source, dtype=np.float64)
    zr_m = np.asarray(zr_stim, dtype=np.float64)
    zc_m = np.asarray(zc_stim, dtype=np.float64)

    n_test, n_src, n_stim = zc_t.shape[0], zc_s.shape[0], zc_m.shape[0]
    if min(n_test, n_src, n_stim) == 0:
        raise ValueError("CAOT needs non-empty target controls, source controls and stimulated cells")
    if zc_t.shape[1] != zc_s.shape[1] or zc_m.shape[1] != zc_s.shape[1]:
        raise ValueError("content dimension mismatch")
    if zr_s.shape[1] != zr_m.shape[1]:
        raise ValueError("response dimension mismatch")

    # (1) target-aware source marginal b
    anchor = zc_t.mean(axis=0, keepdims=True)
    src_sim = cosine_similarity(anchor, zc_s).ravel()
    b = np.exp((src_sim - src_sim.max()) / max(float(source_tau), 1e-8))
    b = np.maximum(b, 1e-12)
    b = b / b.sum()
    a = np.full(n_stim, 1.0 / n_stim, dtype=np.float64)

    # (2) entropic soft OT in the content subspace.
    # The Gram-matrix identity avoids a (n_stim, n_src, dc) broadcast.
    sq_m = (zc_m ** 2).sum(axis=1)[:, None]
    sq_s = (zc_s ** 2).sum(axis=1)[None, :]
    C = np.sqrt(np.maximum(sq_m + sq_s - 2.0 * (zc_m @ zc_s.T), 0.0))
    C = C / (C.max() + 1e-12)
    G = sinkhorn_coupling(a, b, C, reg=ot_reg, use_pot=use_pot)  # (n_stim, n_src)

    col_mass = G.sum(axis=0, keepdims=True).T                     # (n_src, 1)
    stim_resp_bar = (G.T @ zr_m) / np.maximum(col_mass, 1e-12)    # (n_src, dr)
    delta_src = stim_resp_bar - zr_s                               # (n_src, dr)

    # (3) per-cell top-k attention
    S = np.clip((cosine_similarity(zc_t, zc_s) + 1.0) * 0.5, 0.0, 1.0)
    k = min(max(1, int(np.ceil(n_src * float(ratio)))), n_src)
    idx = np.argsort(S, axis=1)[:, -k:]                            # (n_test, k)
    rows = np.arange(n_test)[:, None]
    S_sel = S[rows, idx]
    W_sel = np.exp((S_sel - S_sel.max(axis=1, keepdims=True)) / max(float(attn_tau), 1e-8))
    W_sel = W_sel / (W_sel.sum(axis=1, keepdims=True) + 1e-12)
    W = np.zeros((n_test, n_src), dtype=np.float64)
    W[rows, idx] = W_sel                                           # sparse weights
    delta_pred = W @ delta_src                                     # (n_test, dr)

    # (4) confidence gate
    conf = S_sel.max(axis=1)
    lo, hi = float(conf.min()), float(conf.max())
    floor = float(np.clip(confidence_floor, 0.0, 1.0))
    gate = floor + (1.0 - floor) * (conf - lo) / (hi - lo + 1e-12)

    return {
        "G": G,
        "W": W,
        "topk_idx": idx,
        "source_weight": b,
        "delta_src": delta_src,
        "delta_pred": delta_pred,
        "gate": gate,
        "top_k": k,
    }
