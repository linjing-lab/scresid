"""Evaluation layer (metrics).

This module is deliberately named ``metrics.py`` rather than ``evaluate.py``:
``scresid.evaluate`` is a public callable, and a same-named submodule would
shadow it on the package object as soon as any sibling export is resolved.

Keeps the full visualization suite (PCA / DEG / regression / violin / dotplot)
and reports the quantitative metrics tied to the three innovations:

* top-DEG response correlation and cosine,
* background-gene off-target drift ratio,
* cell-level delta dispersion (zero in the original implementation),
* content-response decorrelation,
* the four R2 scores plus the common top-DEG count.

Robustness fixes over the original code: corrected DEG variable naming, a
single ``rank_genes_groups`` call, a defined ``r_top`` when ``top_genes`` is
absent, gene-name lookups that never KeyError, string-typed condition columns,
and optional-dependency handling for adjustText.
"""

from __future__ import annotations

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

try:  # optional: only affects text placement, never the metrics
    from adjustText import adjust_text
except ImportError:  # pragma: no cover
    adjust_text = None

from .compare import COMPARISON_KEYS, print_comparison
from .config import DEFAULT_KEY_DIC


# --------------------------------------------------------------------- #
# Regression figure
# --------------------------------------------------------------------- #
def draw_reg(eval_adata,
             cell_type,
             reg_type="mean",
             axis_keys=None,
             condition_key="condition",
             text_gene=None,
             top_genes=None,
             save_path=None,
             title=None,
             draw=True,
             if_fig=False,
             fontsize=14):
    """Gene-level mean/variance regression of predicted vs real response.

    Returns ``[r2_all, r2_top]`` or ``[r2_all, r2_top, fig]``. R2 here is the
    squared Pearson correlation (co-variation trend), not regression-residual
    R2, following the convention of the original implementation.
    """
    if axis_keys is None:
        axis_keys = {"x": "predict", "y": "stimulated"}
    df_stim = eval_adata[eval_adata.obs[condition_key] == axis_keys["y"]].to_df()
    df_pred = eval_adata[eval_adata.obs[condition_key] == axis_keys["x"]].to_df()
    if df_stim.shape[0] == 0 or df_pred.shape[0] == 0:
        raise ValueError("regression needs both stimulated and predict groups")

    if reg_type == "mean":
        stim_value = df_stim.mean().values.reshape(-1, 1)
        pred_value = df_pred.mean().values.reshape(-1, 1)
    elif reg_type == "var":
        stim_value = df_stim.var().values.reshape(-1, 1)
        pred_value = df_pred.var().values.reshape(-1, 1)
    else:
        raise ValueError(f"regression type supports mean and var, {reg_type} not support.")

    data = np.hstack((stim_value, pred_value))
    data_df = pd.DataFrame(data, columns=["case", "predict"], index=df_stim.columns)
    fig, ax = plt.subplots()
    sns.set_theme(color_codes=True)
    sns.regplot(x="case", y="predict", data=data_df, ax=ax, scatter_kws={"s": 8})

    case = stim_value.ravel()
    predict = pred_value.ravel()
    var_pos = {g: i for i, g in enumerate(eval_adata.var_names)}

    if text_gene is not None:
        texts = []
        for g in dict.fromkeys(str(x) for x in text_gene):
            j = var_pos.get(g)
            if j is None:
                continue
            texts.append(ax.text(case[j], predict[j], g, fontsize=11, color="black"))
            ax.plot(case[j], predict[j], "o", color="red", markersize=5)
        if texts and adjust_text is not None:
            arrowprops = dict(arrowstyle="->", color="grey", lw=0.5)
            try:
                adjust_text(texts, x=case, y=predict, ax=ax,
                            arrowprops=arrowprops, force_points=(0.0, 0.0))
            except TypeError:  # adjustText renamed / removed some arguments
                adjust_text(texts, x=case, y=predict, ax=ax, arrowprops=arrowprops)

    r_top = 0.0
    if top_genes is not None:
        genes = [g for g in dict.fromkeys(str(x) for x in top_genes) if g in data_df.index]
        if genes:
            data_deg = data_df.loc[genes, :]
            r_top = round(data_deg["case"].corr(data_deg["predict"], method="pearson"), 3)
            r_top = 0.0 if pd.isna(r_top) else r_top
            x_scale = 0.1 * float(np.max(data_df["case"]))
            y_scale = 0.85 * float(np.max(data_df["predict"]))
            ax.text(x_scale, y_scale,
                    s="$R^2_{top\ genes}$=" + str(round(r_top * r_top, 3)),
                    fontsize=fontsize, color="black")

    r = round(data_df["case"].corr(data_df["predict"], method="pearson"), 3)
    r = 0.0 if pd.isna(r) else r
    x_scale = 0.1 * float(np.max(data_df["case"]))
    y_scale = 0.75 * float(np.max(data_df["predict"]))
    ax.text(x_scale, y_scale, s="$R^2_{all\ genes}$=" + str(round(r * r, 3)),
            fontsize=fontsize, color="black")
    plt.title(title if title else "Regression " + reg_type + " of " + str(cell_type))

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if draw:
        plt.show()
    if if_fig:
        return [round(r * r, 3), round(r_top * r_top, 3), fig]
    plt.close(fig)
    return [round(r * r, 3), round(r_top * r_top, 3)]


def _show_or_save(save_dir, filename, show, fig=None):
    """Unified display / persistence for scanpy and matplotlib figures."""
    if show:
        plt.show()
    elif save_dir:
        os.makedirs(save_dir, exist_ok=True)
        target = os.path.join(save_dir, re.sub(r"[^\w.\-]", "_", filename))
        if fig is None:
            fig = plt.gcf()
        fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close("all")


def _safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _dense(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


# --------------------------------------------------------------------- #
# Build the three-group evaluation AnnData
# --------------------------------------------------------------------- #
def make_eval_adata(adata, pred_adata, cell_to_pred, key_dic=None):
    """Concatenate target control + true stimulated + predicted cells.

    The true stimulated cells of the target type are intentionally absent from
    the training set; they are taken from the full dataset purely as ground
    truth for evaluation.
    """
    kd = dict(DEFAULT_KEY_DIC)
    kd.update({str(k): str(v) for k, v in (key_dic or {}).items()})
    ctk, cdk = kd["cell_type_key"], kd["condition_key"]
    ctrl, stim, predk = kd["ctrl_key"], kd["stim_key"], kd["pred_key"]

    ct = np.asarray(adata.obs[ctk]).astype(str)
    cd = np.asarray(adata.obs[cdk]).astype(str)
    parts = []
    for label, mask in (
        (ctrl, (ct == str(cell_to_pred)) & (cd == str(ctrl))),
        (stim, (ct == str(cell_to_pred)) & (cd == str(stim))),
    ):
        p = adata[mask].copy()
        p.obs = pd.DataFrame(
            {cdk: np.repeat(label, p.n_obs),
             ctk: np.repeat(str(cell_to_pred), p.n_obs)},
            index=p.obs_names,
        ).astype(object)
        parts.append(p)

    pp = pred_adata.copy()
    pp.obs = pd.DataFrame(
        {cdk: np.repeat(predk, pp.n_obs),
         ctk: np.repeat(str(cell_to_pred), pp.n_obs)},
        index=pp.obs_names,
    ).astype(object)
    parts.append(pp)

    X = np.vstack([_dense(p.X) for p in parts])
    obs = pd.concat([p.obs for p in parts], axis=0)
    out = sc.AnnData(X=X, obs=obs, var=parts[0].var.copy())
    out.obs_names_make_unique()
    return out


# --------------------------------------------------------------------- #
# Main evaluation
# --------------------------------------------------------------------- #
def evaluate(eval_adata, cell_type, key_dic=None, model=None, pred_adata=None,
             save_dir=None, show=False, top_n=100):
    """Full evaluation. The legacy ``evaluate(adata, cell_type, key_dic)`` works.

    ``pred_adata`` (the return value of ``SCRESID.predict``) supplies the
    innovation-related diagnostics stored in ``obsm`` / ``uns``.
    """
    kd = dict(DEFAULT_KEY_DIC)
    kd.update({str(k): str(v) for k, v in (key_dic or {}).items()})
    cdk = kd["condition_key"]
    ctrl, stim, predk = kd["ctrl_key"], kd["stim_key"], kd["pred_key"]

    adata = eval_adata.copy()
    adata.obs[cdk] = adata.obs[cdk].astype(str).astype(object)
    if ctrl not in set(adata.obs[cdk]):
        raise ValueError(f"evaluation data is missing the control group {ctrl}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    metrics = {"cell_type": str(cell_type)}

    print("PCA cluster results ...")
    try:
        sc.tl.pca(adata)
        sc.pl.pca(adata, color=cdk, frameon=False,
                  title="PCA of " + str(cell_type) + " by Condition", show=False)
        _show_or_save(save_dir, f"{cell_type}_pca.png", show)
    except Exception as exc:  # tiny data / unavailable SVD solver
        print(f"  PCA skipped: {exc}")

    print("Compute rank_genes_groups ...")
    sc.tl.rank_genes_groups(adata, groupby=cdk, reference=ctrl, method="wilcoxon")
    names = adata.uns["rank_genes_groups"]["names"]
    fields = list(names.dtype.names or ())

    def _degs(field):
        if field not in fields:
            return []
        return [str(g) for g in np.asarray(names[field]).ravel()]

    degs_stim = _degs(stim)     # real perturbation vs control
    degs_pred = _degs(predk)    # predicted perturbation vs control
    common = sorted(set(degs_stim[:top_n]) & set(degs_pred[:top_n]))
    metrics["common_top_degs"] = len(common)
    print(f"Common DEGs (top {top_n}): {len(common)}")

    axis_keys = {"x": predk, "y": stim}
    for reg_type in ("mean", "var"):
        res = draw_reg(
            adata, cell_type, reg_type=reg_type, axis_keys=axis_keys,
            condition_key=cdk, text_gene=degs_stim[:10],
            top_genes=degs_stim[:top_n], draw=False, if_fig=True, fontsize=12,
        )
        _show_or_save(save_dir, f"{cell_type}_reg_{reg_type}.png", show, fig=res[2])
        metrics[f"r2_{reg_type}_all"] = res[0]
        metrics[f"r2_{reg_type}_top"] = res[1]
        print(f"R2 {reg_type}: all genes {res[0]}, top genes {res[1]}")

    print("Plot rank_genes_groups / violin / dotplot ...")
    sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, show=False)
    _show_or_save(save_dir, f"{cell_type}_rank_genes.png", show)

    for i, gene in enumerate(degs_stim[:3]):
        if gene not in adata.var_names:
            continue
        sc.pl.violin(adata, keys=gene, groupby=cdk, show=False)
        _show_or_save(save_dir, f"{cell_type}_violin_{i}_{gene}.png", show)

    marker_genes = [g for g in dict.fromkeys(degs_stim[:20]) if g in adata.var_names]
    if marker_genes:
        sc.pl.dotplot(adata, marker_genes, groupby=cdk, show=False)
        _show_or_save(save_dir, f"{cell_type}_dotplot.png", show)

    # ---------------- innovation-related quantitative metrics ---------------- #
    def _mean_vec(cond):
        mask = adata.obs[cdk] == cond
        if not bool(np.any(mask)):
            return None
        return np.asarray(adata[mask].to_df().mean().values, dtype=np.float64)

    mc, ms, mp = _mean_vec(ctrl), _mean_vec(stim), _mean_vec(predk)
    if mc is not None and ms is not None and mp is not None:
        d_true, d_pred = ms - mc, mp - mc
        top = [g for g in dict.fromkeys(degs_stim[:top_n]) if g in adata.var_names]
        top_idx = adata.var_names.get_indexer(top)
        bg_idx = np.setdiff1d(np.arange(adata.n_vars), top_idx)
        metrics["top_response_pearson"] = (
            _safe_corr(d_pred[top_idx], d_true[top_idx]) if top_idx.size else 0.0
        )
        metrics["top_response_cosine"] = float(
            np.dot(d_pred[top_idx], d_true[top_idx]) /
            (np.linalg.norm(d_pred[top_idx]) * np.linalg.norm(d_true[top_idx]) + 1e-12)
        ) if top_idx.size else 0.0
        bg_true = float(np.abs(d_true[bg_idx]).mean()) if bg_idx.size else 0.0
        bg_pred = float(np.abs(d_pred[bg_idx]).mean()) if bg_idx.size else 0.0
        # Smaller is better: only real DEGs should move.
        metrics["offtarget_ratio"] = bg_pred / (bg_true + 1e-12)

    if pred_adata is not None:
        info = dict(pred_adata.uns.get("scresid", {}))
        metrics["cell_level_delta_std"] = float(info.get("delta_std", 0.0))
        metrics["mean_confidence_gate"] = float(info.get("mean_gate", 0.0))
        metrics["n_sources_top_k"] = int(info.get("top_k", 0))
        if ("X_scresid_content" in pred_adata.obsm
                and "X_scresid_response" in pred_adata.obsm):
            Zc = np.asarray(pred_adata.obsm["X_scresid_content"], dtype=np.float64)
            Zr = np.asarray(pred_adata.obsm["X_scresid_response"], dtype=np.float64)
            Zc = Zc - Zc.mean(0, keepdims=True)
            Zr = Zr - Zr.mean(0, keepdims=True)
            C = np.corrcoef(np.concatenate([Zc, Zr], axis=1), rowvar=False)
            C = np.nan_to_num(C)
            metrics["content_response_abs_corr"] = float(
                np.abs(C[: Zc.shape[1], Zc.shape[1]:]).mean()
            )

    _ = model  # reserved extension point
    print("---- scresid metrics ----")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:32s} {v:.4f}")
    return metrics
