"""Innovation III - LOCO-SC: Leave-One-Cell-type-Out Self-Calibration.

Motivation
----------
A VAE computes the perturbation delta in latent space, but every evaluation
metric lives in gene space. A correct latent direction does not guarantee a
correct gene-space direction: the decoder is nonlinear, reconstruction
flattens the per-gene variance spectrum, and a single global delta erases
cell-to-cell response variation.

LOCO-SC closes that gap using **training data only**, in five parts:

(a) ARE - Adaptive Ratio Ensemble. A hand-picked ``ratio`` is clamped by an
    auto-derived fraction capped at ``auto_ratio_cap`` source controls;
    predictions from several ratios are averaged in gene space.
(b) RA  - Response Anchoring. The gene-space response direction is nudged
    toward a cross-cell-type shared response estimated from other types.
(c) VA  - Variance Alignment. The per-gene std spectrum is nudged toward the
    stimulated std spectrum of other types.
(d) LOCO grid search. For each sufficiently large non-target cell type, its
    stimulated cells are temporarily withheld from the prediction sources and
    used as validation labels. The (alpha, var_strength) pair maximizing
    ``0.5*r2_mean_top + 0.5*r2_var_all`` is selected per type, and the medians
    are transferred to the target type. Target stimulated labels are never
    touched, so there is no label leakage.
(e) Safeguard. ``decide_by_loco`` can veto the whole calibration when it does
    not improve the mean LOCO score; ``response_support`` / ``gate_response_X``
    provide a variance-preserving fallback that edits gene means only. The
    safeguard matters because on some datasets variance alignment is a net
    loss and must be switched off rather than applied blindly.

Only NumPy / pandas are used here; torch / scanpy are never imported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import DEFAULT_KEY_DIC

_EPS = 1e-12


# --------------------------------------------------------------------- #
# Metric replicas (same convention as evaluate.draw_reg: pandas .var ddof=1)
# --------------------------------------------------------------------- #
def pearson_r2(x, y):
    """Squared Pearson correlation of two 1-D vectors."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    if den < _EPS:
        return 0.0
    return float((float(x @ y) / den) ** 2)


def metric_r2(stim_X, pred_X, reg_type="mean", gene_idx=None):
    """Replicate the ``case`` vs ``predict`` convention of draw_reg."""
    A = np.asarray(stim_X, dtype=np.float64)
    B = np.asarray(pred_X, dtype=np.float64)
    if gene_idx is not None:
        A, B = A[:, gene_idx], B[:, gene_idx]
    if reg_type == "mean":
        va, vb = A.mean(axis=0), B.mean(axis=0)
    elif reg_type == "var":
        va, vb = A.var(axis=0, ddof=1), B.var(axis=0, ddof=1)
    else:
        raise ValueError("reg_type must be 'mean' or 'var'")
    return pearson_r2(va, vb)


def combined_score(stim_X, pred_X, top_idx, w_top=0.5, w_var=0.5):
    """LOCO objective: top-DEG mean alignment plus all-gene variance alignment."""
    return (
        w_top * metric_r2(stim_X, pred_X, "mean", top_idx)
        + w_var * metric_r2(stim_X, pred_X, "var")
    )


def top_deg_index(ctrl_X, stim_X, top_n=100):
    """Top response genes ranked by |delta mean| (pure NumPy, reproducible)."""
    d = np.abs(np.asarray(stim_X, np.float64).mean(0)
               - np.asarray(ctrl_X, np.float64).mean(0))
    return np.argsort(d)[-min(int(top_n), d.size):]


# --------------------------------------------------------------------- #
# Training-only statistics
# --------------------------------------------------------------------- #
def shared_response(X, cell_types, conditions, cell_type_key_values,
                    ctrl_key, stim_key, exclude=None, robust="median"):
    """Cross-cell-type shared perturbation response ``r_gene``.

    For each (non-excluded) type compute ``mean(stim) - mean(ctrl)`` and then
    take a median / mean across types. The median is robust to one type with
    an outlier response.
    """
    X = np.asarray(X, dtype=np.float64)
    ct = np.asarray(cell_types).astype(str)
    cd = np.asarray(conditions).astype(str)
    rows = []
    for t in cell_type_key_values:
        if exclude is not None and str(t) == str(exclude):
            continue
        m_c = (ct == str(t)) & (cd == str(ctrl_key))
        m_s = (ct == str(t)) & (cd == str(stim_key))
        if int(m_c.sum()) < 2 or int(m_s.sum()) < 2:
            continue
        rows.append(X[m_s].mean(0) - X[m_c].mean(0))
    if not rows:
        raise ValueError("no paired cell types available for the shared response")
    R = np.stack(rows, axis=0)
    return np.median(R, axis=0) if robust == "median" else R.mean(axis=0)


def variance_target(X, cell_types, conditions, types, stim_key, exclude=None,
                    lo=0.5, hi=2.0):
    """Median per-gene std of other types' stimulated cells."""
    X = np.asarray(X, dtype=np.float64)
    ct = np.asarray(cell_types).astype(str)
    cd = np.asarray(conditions).astype(str)
    rows = []
    for t in types:
        if exclude is not None and str(t) == str(exclude):
            continue
        m = (ct == str(t)) & (cd == str(stim_key))
        if int(m.sum()) >= 3:
            rows.append(X[m].std(axis=0, ddof=1))
    if not rows:
        raise ValueError("no stimulated cells available for the variance target")
    return np.median(np.stack(rows, axis=0), axis=0)


# --------------------------------------------------------------------- #
# Calibration operators
# --------------------------------------------------------------------- #
def calibrate_X(pred0_X, ctrl_X, r_gene, alpha=0.0, var_target=None,
                var_strength=0.0, clip_ratio=(0.5, 2.0)):
    """Anchor the gene-space response to ``r_gene`` and align the variance.

    ``alpha=0, var_strength=0`` is an exact identity (no side effects).

    d_hidden = mean(pred0) - mean(ctrl)
    r_hat    = r_gene * ||d_hidden|| / ||r_gene||
    d_final  = (1-a) * d_hidden + a * r_hat
    pred_a   = pred0 + (d_final - d_hidden)
    s_g      = clip(var_target / std(pred_a), lo, hi)
    pred     = mean(pred_a) + (1-v)*resid + v*s_g*resid
    """
    P0 = np.asarray(pred0_X, dtype=np.float64)
    C = np.asarray(ctrl_X, dtype=np.float64)
    if P0.shape != C.shape:
        raise ValueError(f"pred0 {P0.shape} and ctrl {C.shape} shape mismatch")
    alpha = float(np.clip(alpha, 0.0, 1.0))
    v = float(np.clip(var_strength, 0.0, 1.0))
    if alpha == 0.0 and v == 0.0:
        return P0.copy()

    P = P0.copy()
    if alpha > 0.0 and r_gene is not None:
        d_hidden = P.mean(0) - C.mean(0)
        r = np.asarray(r_gene, dtype=np.float64)
        nr = np.linalg.norm(r)
        if nr > _EPS:
            r_hat = r * (np.linalg.norm(d_hidden) / (nr + _EPS))
            P = P + alpha * (r_hat - d_hidden)[None, :]

    if v > 0.0 and var_target is not None:
        mu = P.mean(0, keepdims=True)
        resid = P - mu
        cur = P.std(axis=0, ddof=1)
        s = np.asarray(var_target, dtype=np.float64) / (cur + _EPS)
        s = np.clip(np.nan_to_num(s, nan=1.0, posinf=1.0), clip_ratio[0], clip_ratio[1])
        P = mu + (1.0 - v) * resid + v * resid * s[None, :]

    return np.clip(P, 0.0, None)


def grid_search(stim_X, pred0_X, ctrl_X, r_gene, var_target, top_idx,
                alphas=(0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9),
                var_strengths=(0.0, 0.3, 0.6, 1.0),
                w_top=0.5, w_var=0.5):
    """Maximize ``combined_score`` over the (alpha, var_strength) grid.

    Returns ``(best_alpha, best_var_strength, best_score, table)``.
    """
    best = (0.0, 0.0, -np.inf)
    table = []
    for a in alphas:
        for v in var_strengths:
            X = calibrate_X(pred0_X, ctrl_X, r_gene, a, var_target, v)
            s = combined_score(stim_X, X, top_idx, w_top, w_var)
            row = {
                "alpha": float(a),
                "var_strength": float(v),
                "score": float(s),
                "r2_mean_top": metric_r2(stim_X, X, "mean", top_idx),
                "r2_var_all": metric_r2(stim_X, X, "var"),
            }
            table.append(row)
            if s > best[2]:
                best = (float(a), float(v), float(s))
    return (*best, table)


def aggregate_grid(results):
    """Aggregate per-type optima into one robust (alpha, var_strength) pair."""
    if not results:
        return 0.0, 0.0
    a = np.median([r["alpha"] for r in results])
    v = np.median([r["var_strength"] for r in results])
    return float(a), float(v)


def auto_ratio(n_source_ctrl, base=0.5, cap=256):
    """Adaptive ratio: many source controls do not need a literal 50%.

    Guarantees top-k in ``[8, cap]``; small datasets participate fully.
    """
    n = max(int(n_source_ctrl), 1)
    k = min(max(8, int(np.ceil(base * n))), min(cap, n))
    return float(k / n)


def ensemble_X(pred_list):
    """Mean-ensemble several ratio / seed predictions in gene space."""
    if not pred_list:
        raise ValueError("pred_list is empty")
    return np.mean(np.stack([np.asarray(p, np.float64) for p in pred_list], axis=0), axis=0)


# --------------------------------------------------------------------- #
# Variance-preserving fallback operators (safeguard kit)
# --------------------------------------------------------------------- #
def response_support(X, cell_types, conditions, types, ctrl_key, stim_key,
                     exclude=None, top_n=300, power=1.0):
    """Estimate *which genes respond* from other types, without using direction.

    A LOCO alpha of 0 means the cross-type response **direction** is not
    transferable, but the **support set** of responsive genes can still be a
    shared prior. Only the rank of |delta mean| is used, so no target
    stimulated label leaks in.
    """
    X = np.asarray(X, dtype=np.float64)
    ct = np.asarray(cell_types).astype(str)
    cd = np.asarray(conditions).astype(str)
    n_genes = X.shape[1]
    rows = []
    for t in types:
        if exclude is not None and str(t) == str(exclude):
            continue
        m_c = (ct == str(t)) & (cd == str(ctrl_key))
        m_s = (ct == str(t)) & (cd == str(stim_key))
        if int(m_c.sum()) < 2 or int(m_s.sum()) < 2:
            continue
        d = np.abs(X[m_s].mean(0) - X[m_c].mean(0))
        k = min(int(top_n), n_genes)
        idx = np.argsort(d)[-k:]
        mask = np.zeros(n_genes, dtype=np.float64)
        rank = np.argsort(np.argsort(-d[idx]))   # 0 = largest |d|
        mask[idx] = 1.0 - rank / max(k - 1, 1)
        rows.append(mask)
    if not rows:
        return np.ones(n_genes, dtype=np.float64)
    m = np.mean(np.stack(rows, axis=0), axis=0)
    return np.clip(m ** float(power), 0.0, 1.0)


def gate_response_X(pred0_X, ctrl_X, support, center_strength=0.0, gate=0.0):
    """Remove a global decoder shift and shrink background-gene responses.

    Only the per-gene mean is replaced; per-sample residuals are untouched,
    therefore ``r2_var_*`` is strictly unchanged. Useful when variance
    alignment was rejected by the LOCO safeguard.
    """
    P0 = np.asarray(pred0_X, dtype=np.float64)
    C = np.asarray(ctrl_X, dtype=np.float64)
    if P0.shape != C.shape:
        raise ValueError(f"pred0 {P0.shape} and ctrl {C.shape} shape mismatch")
    c = float(np.clip(center_strength, 0.0, 1.0))
    g = float(np.clip(gate, 0.0, 1.0))
    mu0 = P0.mean(0, keepdims=True)
    if c == 0.0 and g == 0.0:
        return P0.copy()

    d = mu0.ravel() - C.mean(0).ravel()
    d2 = d - c * np.median(d)
    m = np.asarray(support, dtype=np.float64).ravel()
    if m.size != d.size:
        raise ValueError("support length does not match the number of genes")
    d2 = d2 * ((1.0 - g) + g * m)
    new_mu = np.clip(mu0.ravel() + (d2 - d), 0.0, None)
    return P0 - mu0 + new_mu[None, :]


def decide_by_loco(scores_off, scores_on, margin=0.005):
    """Veto the calibration unless it improves the *mean* LOCO score.

    Averaging a median across types and then applying it unconditionally can
    hurt a well-aligned dataset. Every post-processing step must first prove
    a positive gain on leave-one-cell-type-out validation.
    """
    off = np.asarray(scores_off, dtype=np.float64)
    on = np.asarray(scores_on, dtype=np.float64)
    if off.size == 0 or on.size == 0 or off.size != on.size:
        return False, 0.0
    delta = float(np.mean(on - off))
    return bool(delta > margin), delta


# --------------------------------------------------------------------- #
# Array / obs helpers (duck-typed: works for AnnData and lightweight stubs)
# --------------------------------------------------------------------- #
def obs_arrays(adata, key_dic):
    """Return string cell-type / condition vectors."""
    ct = np.asarray(adata.obs[key_dic["cell_type_key"]]).astype(str)
    cd = np.asarray(adata.obs[key_dic["condition_key"]]).astype(str)
    return ct, cd


def matrix(adata):
    """Dense float64 expression matrix, sparse-aware."""
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float64)


@dataclass
class CalibrationPlan:
    """Transferable result of a LOCO self-calibration run."""

    alpha: float = 0.0
    var_strength: float = 0.0
    r_gene: np.ndarray = field(default_factory=lambda: np.zeros(0))
    var_target: np.ndarray = field(default_factory=lambda: np.zeros(0))
    support: np.ndarray = field(default_factory=lambda: np.zeros(0))
    accepted: bool = True
    mean_delta: float = 0.0
    ratios: Sequence[float] = ()
    details: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "alpha": float(self.alpha),
            "var_strength": float(self.var_strength),
            "r_gene": np.asarray(self.r_gene, dtype=np.float64),
            "var_target": np.asarray(self.var_target, dtype=np.float64),
            "support": np.asarray(self.support, dtype=np.float64),
            "accepted": bool(self.accepted),
            "mean_delta": float(self.mean_delta),
            "ratios": tuple(float(r) for r in self.ratios),
            "details": [dict(d) for d in self.details],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationPlan":
        return cls(
            alpha=float(d.get("alpha", 0.0)),
            var_strength=float(d.get("var_strength", 0.0)),
            r_gene=np.asarray(d.get("r_gene", []), dtype=np.float64),
            var_target=np.asarray(d.get("var_target", []), dtype=np.float64),
            support=np.asarray(d.get("support", []), dtype=np.float64),
            accepted=bool(d.get("accepted", True)),
            mean_delta=float(d.get("mean_delta", 0.0)),
            ratios=tuple(d.get("ratios", ())),
            details=list(d.get("details", [])),
        )


def _target_statistics(X_all, ct, cd, source_types, ctrl, stim,
                       cell_to_pred, verbose):
    """Training-only statistics for the target type, with a safe fallback.

    ``source_types`` already excludes the target type, so even if the caller
    left the target's stimulated cells inside ``train`` they can never leak
    into the shared response / variance prior.
    """
    try:
        r_gene = shared_response(
            X_all, ct, cd, source_types, ctrl, stim, exclude=cell_to_pred)
        var_target = variance_target(
            X_all, ct, cd, source_types, stim, exclude=cell_to_pred)
        ok = True
    except ValueError as exc:
        if verbose:
            print(f"[LOCO] target statistics unavailable: {exc}")
        r_gene = np.zeros(X_all.shape[1], dtype=np.float64)
        var_target = np.ones(X_all.shape[1], dtype=np.float64)
        ok = False
    support = response_support(
        X_all, ct, cd, source_types, ctrl, stim,
        exclude=cell_to_pred, top_n=300)
    return r_gene, var_target, support, ok


def loco_calibrate(model, train_adata, cell_to_pred, key_dic=None,
                   ratio=None, ratios=(), verbose=True) -> CalibrationPlan:
    """Run leave-one-cell-type-out self-calibration and return a plan.

    For every large non-target type ``t``:
      * build a training view without ``t``'s stimulated cells,
      * predict ``t``'s controls through CAOT (``run_loco=False`` to avoid
        recursion),
      * use ``t``'s real stimulated cells as LOCO validation labels,
      * grid-search (alpha, var_strength) and report the best row.

    The medians of the per-type optima, plus statistics estimated with the
    target type excluded, form the plan that is transferred to the target.

    Feasibility
    -----------
    A leave-one-out fold needs at least **two** non-target cell types with
    stimulated cells: after withholding one type's stimulated cells, the
    remaining stimulated cells are the only source CAOT can transfer from. A
    two-cell-type dataset has exactly one such type, so LOCO is mathematically
    impossible. In that case this function prints a diagnostic and returns a
    no-op plan (``alpha=0, var_strength=0, accepted=False``) whose statistics
    are still populated, so a user can construct a plan manually if desired.
    No exception is raised and the baseline prediction stays intact.
    """
    cfg = model.cfg
    kd = cfg.override(key_dic)
    ct, cd = obs_arrays(train_adata, kd)
    ctrl, stim = kd["ctrl_key"], kd["stim_key"]
    types = [str(t) for t in pd.unique(ct)]
    # The target type's stimulated cells are the answer, never a source.
    # Excluding them here also makes the calibration correct when the caller
    # did not withhold them from train.
    source_types = [t for t in types if t != str(cell_to_pred)]

    counts = pd.Series(ct).value_counts()
    candidates = [
        str(t) for t in counts.index
        if str(t) != str(cell_to_pred)
        and int(counts[t]) >= int(cfg.loco_min_cells)
        and int(((ct == str(t)) & (cd == ctrl)).sum()) >= 2
        and int(((ct == str(t)) & (cd == stim)).sum()) >= 2
    ]

    X_all = matrix(train_adata)
    results: List[dict] = []
    scores_off, scores_on = [], []

    # ---- feasibility filter: a fold must leave another source type that
    #      still has stimulated cells for CAOT to transfer from ----
    usable = []
    for t in candidates:
        keep = ~((ct == t) & (cd == stim))
        remaining = [
            u for u in source_types
            if u != t and int(((ct[keep] == u) & (cd[keep] == stim)).sum()) > 0
        ]
        if not remaining:
            continue  # no stimulated source left after withholding t
        if int(((ct[keep] != t) & (cd[keep] == ctrl)).sum()) == 0:
            continue  # no other-type control source left
        usable.append(t)

    n_stim_types = sum(
        1 for t in source_types
        if int(((ct == t) & (cd == stim)).sum()) > 0
    )
    if not usable:
        if verbose:
            print(
                f"[LOCO] skipped: {n_stim_types} non-target cell type(s) with "
                "stimulated cells; leave-one-out needs >= 2"
            )
            print("[LOCO] fallback -> alpha=0.00, var_strength=0.00 "
                  "(no-op plan, baseline preserved)")
        r_gene, var_target, support, _ = _target_statistics(
            X_all, ct, cd, source_types, ctrl, stim, cell_to_pred, verbose)
        return CalibrationPlan(
            alpha=0.0, var_strength=0.0,
            r_gene=r_gene, var_target=var_target, support=support,
            accepted=False, mean_delta=0.0,
            ratios=tuple(float(r) for r in (ratios or ())),
            details=[],
        )

    for t in usable:
        keep = ~((ct == t) & (cd == stim))
        sub = train_adata[keep].copy()
        pred_t = model.predict(
            sub, cell_to_pred=t, key_dic=kd, ratio=ratio,
            ensemble=False, run_loco=False, clamp_ratio=False,
            verbose=verbose,
        )

        m_c = (ct == t) & (cd == ctrl)
        m_s = (ct == t) & (cd == stim)
        ctrl_X = matrix(train_adata[m_c])
        stim_X = matrix(train_adata[m_s])
        pred0_X = matrix(pred_t)
        if pred0_X.shape != ctrl_X.shape:
            raise ValueError(
                f"LOCO shape mismatch for {t}: pred {pred0_X.shape}, ctrl {ctrl_X.shape}"
            )

        try:
            r_gene = shared_response(
                X_all, ct, cd, source_types, ctrl, stim, exclude=t)
            var_target = variance_target(
                X_all, ct, cd, source_types, stim, exclude=t)
        except ValueError as exc:
            if verbose:
                print(f"[LOCO] {t:<18} skipped: {exc}")
            continue
        top_idx = top_deg_index(ctrl_X, stim_X, int(cfg.loco_top_n))

        a, v, score, table = grid_search(
            stim_X, pred0_X, ctrl_X, r_gene, var_target, top_idx,
            alphas=cfg.alphas, var_strengths=cfg.var_strengths,
            w_top=cfg.loco_w_top, w_var=cfg.loco_w_var,
        )
        best = max(table, key=lambda row: row["score"])
        results.append({
            "cell_type": t,
            "alpha": a,
            "var_strength": v,
            "score": score,
            "r2_mean_top": best["r2_mean_top"],
            "r2_var_all": best["r2_var_all"],
        })
        scores_off.append(combined_score(
            stim_X, pred0_X, top_idx, cfg.loco_w_top, cfg.loco_w_var))
        scores_on.append(best["score"])

        if verbose:
            print(
                f"[LOCO] {t:<18} alpha={a:.2f} var={v:.2f} "
                f"mean_top={best['r2_mean_top']:.4f} "
                f"var_all={best['r2_var_all']:.4f}"
            )

    alpha, var_strength = aggregate_grid(results)
    if verbose:
        print(f"[LOCO] aggregated -> alpha={alpha:.2f}, var_strength={var_strength:.2f}")

    r_gene_target, var_target_target, support, stats_ok = _target_statistics(
        X_all, ct, cd, source_types, ctrl, stim, cell_to_pred, verbose)
    if not stats_ok or not results:
        alpha, var_strength = 0.0, 0.0
        if verbose and not results:
            print("[LOCO] no usable folds -> alpha=0.00, var_strength=0.00 "
                  "(no-op plan, baseline preserved)")
    accepted, mean_delta = decide_by_loco(scores_off, scores_on)

    return CalibrationPlan(
        alpha=alpha,
        var_strength=var_strength,
        r_gene=r_gene_target,
        var_target=var_target_target,
        support=support,
        accepted=accepted,
        mean_delta=mean_delta,
        ratios=tuple(float(r) for r in (ratios or ())),
        details=results,
    )


# --------------------------------------------------------------------- #
# Argument normalization: the plan and the key_dic are both dicts, so their
# positional order must not be able to break the call.
# --------------------------------------------------------------------- #
_PLAN_KEYS = frozenset({
    "alpha", "var_strength", "r_gene", "var_target", "support",
    "accepted", "mean_delta", "ratios", "details",
})
_KEY_DIC_KEYS = frozenset(DEFAULT_KEY_DIC)


def _coerce_plan(raw, pred_adata):
    """Return a ``CalibrationPlan`` from an instance, a dict, or ``None``."""
    if isinstance(raw, CalibrationPlan):
        return raw
    if raw is None:
        info = dict(pred_adata.uns.get("scresid", {}))
        raw = info.get("calibration")
        if raw is None:
            raise ValueError(
                "no calibration plan in pred_adata.uns['scresid']; run "
                "predict with run_loco=True or pass plan= explicitly"
            )
    if isinstance(raw, dict):
        return CalibrationPlan.from_dict(raw)
    raise TypeError(f"unsupported calibration plan type: {type(raw).__name__}")


def _normalize_calibration_args(plan, key_dic, pred_adata):
    """Resolve ``plan`` / ``key_dic`` regardless of their positional order.

    All of these are accepted::

        apply_calibration(pred, train, ct, key_dic)
        apply_calibration(pred, train, ct)
        apply_calibration(pred, train, ct, plan, key_dic)
        apply_calibration(pred, train, ct, plan=plan, key_dic=key_dic)

    A dict is classified by its keys: a key_dic contains obs column names
    (``condition_key``, ``ctrl_key``, ...), a plan dict contains ``alpha``,
    ``var_strength``, ``r_gene``, ... When the plan is omitted it is loaded
    from ``pred_adata.uns['scresid']['calibration']``.
    """
    info = dict(pred_adata.uns.get("scresid", {}))
    plan_obj, kd_dict = None, None
    for arg in (plan, key_dic):
        if arg is None:
            continue
        if isinstance(arg, CalibrationPlan):
            plan_obj = arg
        elif isinstance(arg, dict):
            keys = set(arg)
            if keys & _KEY_DIC_KEYS:
                kd_dict = dict(arg)
            elif keys & _PLAN_KEYS:
                plan_obj = CalibrationPlan.from_dict(arg)
            else:
                # Unknown dict: treat it as a key_dic override, not as a plan.
                kd_dict = dict(arg)
        else:
            raise TypeError(
                f"plan/key_dic must be a dict or CalibrationPlan, got "
                f"{type(arg).__name__}"
            )
    if plan_obj is None:
        plan_obj = _coerce_plan(None, pred_adata)
    if kd_dict is None:
        kd_dict = dict(info.get("key_dic") or DEFAULT_KEY_DIC)
    return plan_obj, kd_dict, info


def apply_calibration(pred_adata, train_adata, cell_to_pred, plan=None,
                      key_dic=None, guard=False):
    """Apply a LOCO plan to a prediction, returning a new AnnData-like object.

    ``plan`` may be a ``CalibrationPlan``, the plan dict stored in
    ``uns['scresid']['calibration']``, or ``None`` (auto-load from ``uns``).
    For convenience a ``key_dic`` dict passed in the ``plan`` slot is also
    recognized, so ``apply_calibration(pred, train, ct, key_dic)`` works.

    ``guard=True`` disables the calibration when ``decide_by_loco`` rejected
    it, keeping the baseline prediction intact. Only ``X`` is changed, so the
    latent diagnostics in ``obsm`` / ``uns`` (delta std, gate, content-response
    correlation) stay identical across baseline and calibrated evaluations.
    """
    plan, kd, info = _normalize_calibration_args(plan, key_dic, pred_adata)

    ct, cd = obs_arrays(train_adata, kd)
    m = (ct == str(cell_to_pred)) & (cd == str(kd["ctrl_key"]))
    ctrl_part = train_adata[m].copy()

    P0 = matrix(pred_adata)
    ctrl_X = matrix(ctrl_part)
    try:
        idx = ctrl_part.obs_names.get_indexer(pred_adata.obs_names)
        if np.all(idx >= 0):
            ctrl_X = ctrl_X[idx]
    except (AttributeError, TypeError):
        pass
    if ctrl_X.shape != P0.shape:
        raise ValueError(
            f"control {ctrl_X.shape} and prediction {P0.shape} shape mismatch"
        )

    alpha = plan.alpha
    var_strength = plan.var_strength
    if guard and not plan.accepted:
        alpha, var_strength = 0.0, 0.0

    X = calibrate_X(P0, ctrl_X, plan.r_gene, alpha,
                    plan.var_target, var_strength)
    out = pred_adata.copy()
    out.X = np.asarray(X, dtype=np.float32)
    info["calibrated"] = bool(alpha > 0.0 or var_strength > 0.0)
    info["calibration_alpha"] = float(alpha)
    info["calibration_var_strength"] = float(var_strength)
    out.uns["scresid"] = info
    return out


# --------------------------------------------------------------------- #
# Small convenience wrapper used by the benchmark runner
# --------------------------------------------------------------------- #
def support_gate_calibration(pred_adata, train_adata, cell_to_pred, plan=None,
                             key_dic=None, center_strength=0.0, gate=0.0):
    """Variance-preserving alternative: never touches ``r2_var_*``.

    Accepts the same flexible ``plan`` / ``key_dic`` argument forms as
    :func:`apply_calibration`.
    """
    plan, kd, info = _normalize_calibration_args(plan, key_dic, pred_adata)
    ct, cd = obs_arrays(train_adata, kd)
    m = (ct == str(cell_to_pred)) & (cd == str(kd["ctrl_key"]))
    ctrl_X = matrix(train_adata[m].copy())
    X = gate_response_X(matrix(pred_adata), ctrl_X, plan.support,
                        center_strength, gate)
    out = pred_adata.copy()
    out.X = np.asarray(X, dtype=np.float32)
    out.uns["scresid"] = info
    return out
