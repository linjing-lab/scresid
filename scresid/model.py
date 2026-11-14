"""Main model: SCRESID, integrating three innovations.

Innovation I -- DC-RL (Disentangled Condition-Residual Latent)
    The latent vector is explicitly split into ``z = [z_content | z_response]``
    and trained with three light regularizers that use only ``obs`` labels:
      * intra-type compactness / inter-type separation / cross-condition
        invariance  -> ``content`` encodes "who the cell is";
      * control response -> 0, same-type stimulated responses consistent
        -> ``response`` encodes "how much it was perturbed";
      * content-response decorrelation -> the two segments do not leak.
    At inference only the response segment is transferred; the content segment
    is preserved verbatim, which suppresses identity-driven off-target drift.

Innovation II -- CAOT (see transport.py)
    Target-aware marginal + soft OT + per-cell top-k attention + confidence
    gate produce a cell-level ``delta_pred``, replacing the hard-argmax global
    delta of the original implementation.

Innovation III -- LOCO-SC (see calibration.py)
    Adaptive ratio ensemble plus leave-one-cell-type-out self-calibration in
    gene space (response anchoring + variance alignment), with a safeguard
    that can veto a harmful calibration.

Torch / scanpy / tqdm are expected to be pre-installed; this package never
installs or downloads them. tqdm has a tiny built-in stand-in so training
still runs in minimal environments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - keeps training usable without tqdm
    class tqdm:
        """Minimal stand-in that preserves the epoch description text."""

        def __init__(self, iterable=None, disable=False, **kwargs):
            self.iterable = list(iterable or [])
            self.disable = bool(disable)
            self._desc = ""

        def __iter__(self):
            for item in self.iterable:
                yield item
            if not self.disable and self._desc:
                print(self._desc)

        def set_description(self, text):
            self._desc = text

from .calibration import auto_ratio, ensemble_X, loco_calibrate
from .config import SCRESIDConfig
from .data import PerturbDataset, encode_labels
from .modules import EfficientDecoder, EfficientEncoder
from .transport import caot_transfer


class SCRESID(nn.Module):
    def __init__(self, config=None, device=None, **kwargs):
        super().__init__()
        if isinstance(config, SCRESIDConfig):
            if kwargs:
                raise TypeError("extra keyword arguments are not allowed with SCRESIDConfig")
            self.cfg = config
        else:
            params = dict(config or {})
            params.update(kwargs)
            self.cfg = SCRESIDConfig(**params)

        if device is None:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            dev = torch.device("cpu")
        self.device = dev

        cfg = self.cfg
        self.encoder = EfficientEncoder(
            cfg.input_dim, cfg.hidden_dim, cfg.latent_dim,
            n_blocks=cfg.n_res_blocks, dropout=cfg.dropout,
        )
        self.decoder = EfficientDecoder(
            cfg.latent_dim, cfg.hidden_dim, cfg.input_dim,
            n_blocks=cfg.n_res_blocks, dropout=cfg.dropout,
        )
        self.to(self.device)
        self.history = []
        self._preflight = {}
        self.last_ratios = ()
        self._main_ratios = ()

    # ------------------------------------------------------------------ #
    # Latent space (Innovation I)
    # ------------------------------------------------------------------ #
    def split_z(self, z):
        """Split ``(..., latent_dim)`` into ``(z_content, z_response)``."""
        c = self.cfg.content_dim
        return z[..., :c], z[..., c:]

    def join_z(self, z_content, z_response):
        return torch.cat([z_content, z_response], dim=-1)

    def encode(self, x, use_mean=False):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        logvar = torch.clamp(logvar, -8.0, 8.0)
        if use_mean:
            return mu, mu, logvar
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std, mu, logvar

    def forward(self, x, add_noise=False, use_mean=False):
        """Return reconstruction, latent segments, and per-sample loss terms."""
        if add_noise and self.training:
            x_in = x + torch.randn_like(x) * self.cfg.noise_rate
        else:
            x_in = x
        z, mu, logvar = self.encode(x_in, use_mean=use_mean)
        z_c, z_r = self.split_z(z)
        x_hat = self.decoder(z)
        kl = 0.5 * torch.sum(-1.0 - logvar + mu.pow(2) + logvar.exp(), dim=-1)
        rec = torch.sum((x - x_hat) ** 2, dim=-1)
        return {
            "x_hat": x_hat, "z": z, "z_c": z_c, "z_r": z_r,
            "mu": mu, "logvar": logvar, "rec": rec, "kl": kl,
        }

    def disentangle_loss(self, z_c, z_r, cell_type, is_stim):
        """Batch prototype contrast + response residual + decorrelation."""
        cfg = self.cfg
        device = z_c.device
        cell_type = cell_type.long().reshape(-1)
        is_stim = is_stim.float().reshape(-1)
        ctrl = is_stim < 0.5
        stim = ~ctrl
        zero = z_c.sum() * 0.0  # keeps the graph connected for empty branches

        # (a) control has no response; same-type stimulated responses agree
        l_resp_ctrl = (z_r[ctrl] ** 2).mean() if bool(ctrl.any()) else zero
        resp_cons = []
        for t in torch.unique(cell_type):
            m = (cell_type == t) & stim
            r = z_r[m]
            if r.shape[0] > 1:
                resp_cons.append(((r - r.mean(0, keepdim=True)) ** 2).mean())
        l_resp_cons = torch.stack(resp_cons).mean() if resp_cons else zero

        # (b) content: intra-type compact, inter-type separated, condition invariant
        protos, intra, cond_gap = [], [], []
        for t in torch.unique(cell_type):
            m = cell_type == t
            zc_t = z_c[m]
            proto = zc_t.mean(0)
            protos.append(proto)
            if zc_t.shape[0] > 1:
                intra.append(((zc_t - proto) ** 2).mean())
            a, b = z_c[m & ctrl], z_c[m & stim]
            if a.shape[0] > 0 and b.shape[0] > 0:
                cond_gap.append(((a.mean(0) - b.mean(0)) ** 2).mean())
        l_intra = torch.stack(intra).mean() if intra else zero
        l_cond = torch.stack(cond_gap).mean() if cond_gap else zero
        if len(protos) > 1:
            P = torch.stack(protos)
            D = (P[:, None, :] - P[None, :, :]).norm(dim=-1)
            off = ~torch.eye(P.shape[0], dtype=torch.bool, device=device)
            l_inter = F.relu(cfg.content_margin - D[off]).mean()
        else:
            l_inter = zero

        # (c) content / response decorrelation
        n = z_c.shape[0]
        if n > 1:
            zc0 = z_c - z_c.mean(0, keepdim=True)
            zr0 = z_r - z_r.mean(0, keepdim=True)
            cov = zc0.T @ zr0 / (n - 1)
            l_decorr = (cov ** 2).mean()
        else:
            l_decorr = zero

        parts = {
            "content": l_intra + l_inter + l_cond,
            "resp": l_resp_ctrl + l_resp_cons,
            "decorr": l_decorr,
        }
        detail = {
            "intra": float(l_intra.detach()),
            "inter": float(l_inter.detach()),
            "cond": float(l_cond.detach()),
            "resp_ctrl": float(l_resp_ctrl.detach()),
            "resp_cons": float(l_resp_cons.detach()),
            "decorr": float(l_decorr.detach()),
        }
        return parts, detail

    # ------------------------------------------------------------------ #
    # Preflight for Innovation III (prints the benchmark planning line)
    # ------------------------------------------------------------------ #
    def _run_preflight(self, train_adata, kd, verbose=True):
        ct = np.asarray(train_adata.obs[kd["cell_type_key"]]).astype(str)
        cd = np.asarray(train_adata.obs[kd["condition_key"]]).astype(str)
        ctrl, stim = kd["ctrl_key"], kd["stim_key"]
        counts = pd.Series(ct).value_counts()
        missing = [
            str(t) for t in counts.index
            if not np.any((ct == str(t)) & (cd == stim))
        ]
        # ``predict`` excludes exactly one cell type from its source controls.
        # Mirror that rule here so the printed planning line matches the
        # ratios actually used. When no type is fully withheld (some datasets
        # keep the target's stimulated cells in train), fall back to the
        # largest control group as the assumed target.
        ctrl_counts = pd.Series(ct[cd == ctrl]).value_counts()
        if len(missing) == 1:
            target = str(missing[0])
        elif len(ctrl_counts) > 1:
            target = str(ctrl_counts.index[0])
        else:
            target = None
        if target is not None:
            n_src = int(((ct != target) & (cd == ctrl)).sum())
        else:
            n_src = int((cd == ctrl).sum())

        cfg = self.cfg
        auto = round(float(auto_ratio(n_src, cfg.auto_ratio_base,
                                      cfg.auto_ratio_cap)), 4)
        base = round(float(cfg.ratio), 4)
        ratios = tuple(sorted(set([base, auto])))
        if verbose:
            print(
                "source control cells = {} | auto ratio = {:.4f} | "
                "ensemble = [{}]".format(
                    n_src, auto, ", ".join(f"{r:g}" for r in ratios))
            )
        return {
            "target": target,
            "n_source_control": n_src,
            "auto_ratio": auto,
            "ratios": ratios,
        }

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train_model(self, train_adata, epochs=100, batch_size=128, lr=None,
                    key_dic=None, verbose=True):
        cfg = self.cfg
        kd = cfg.override(key_dic)
        self._preflight = self._run_preflight(train_adata, kd, verbose=verbose)
        codes, is_stim = encode_labels(train_adata, kd)
        loader = DataLoader(
            PerturbDataset(train_adata, codes, is_stim),
            batch_size=batch_size, shuffle=True, drop_last=False,
        )
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=cfg.lr if lr is None else lr,
            weight_decay=cfg.weight_decay, betas=(0.9, 0.999),
        )
        self.train()
        self.history = []
        pbar = tqdm(range(int(epochs)), disable=not verbose)
        for epoch in pbar:
            acc = {k: 0.0 for k in
                   ("loss", "rec", "kl", "content", "resp", "decorr",
                    "intra", "inter", "cond")}
            n_batches = 0
            for batch in loader:
                x = batch["x"].to(self.device).float()
                ct = batch["cell_type"].to(self.device).long()
                st = batch["is_stim"].to(self.device).float()

                out = self.forward(x, add_noise=True)
                parts, detail = self.disentangle_loss(out["z_c"], out["z_r"], ct, st)
                loss = (
                    out["rec"].mean()
                    + cfg.kl_weight * out["kl"].mean()
                    + cfg.lambda_content * parts["content"]
                    + cfg.lambda_resp * parts["resp"]
                    + cfg.lambda_decorr * parts["decorr"]
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), cfg.grad_clip)
                optimizer.step()

                n_batches += 1
                acc["loss"] += float(loss.detach())
                acc["rec"] += float(out["rec"].mean().detach())
                acc["kl"] += float(out["kl"].mean().detach())
                acc["content"] += float(parts["content"].detach())
                acc["resp"] += float(parts["resp"].detach())
                acc["decorr"] += float(parts["decorr"].detach())
                acc["intra"] += detail["intra"]
                acc["inter"] += detail["inter"]
                acc["cond"] += detail["cond"]

            row = {k: v / max(n_batches, 1) for k, v in acc.items()}
            row["epoch"] = epoch
            self.history.append(row)
            if verbose:
                pbar.set_description(
                    "Epoch {:>3d} | loss {:.3f} rec {:.3f} content {:.3f} resp {:.3f}".format(
                        epoch, row["loss"], row["rec"], row["content"], row["resp"])
                )
        self.eval()
        return self.history

    # Compatible name used by the driving script
    train_SCRESID = train_model

    # ------------------------------------------------------------------ #
    # Latent export
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def latent(self, adata, use_mean=True):
        """Inference uses ``mu`` (no reparameterization noise)."""
        self.eval()
        X = torch.as_tensor(
            np.asarray(adata.to_df().values, dtype=np.float32),
            dtype=torch.float32, device=self.device,
        )
        z, _, _ = self.encode(X, use_mean=use_mean)
        return z.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def latent_adata(self, adata, use_mean=True):
        Z = self.latent(adata, use_mean=use_mean)
        c = self.cfg.content_dim
        out = sc.AnnData(X=Z, obs=adata.obs.copy())
        out.obsm["X_scresid_content"] = Z[:, :c].copy()
        out.obsm["X_scresid_response"] = Z[:, c:].copy()
        return out

    # ------------------------------------------------------------------ #
    # Prediction: CAOT + adaptive ensemble + LOCO-SC
    # ------------------------------------------------------------------ #
    def _resolve_ratios(self, requested, n_src, do_ensemble, clamp_ratio=True):
        """Clamp a user ratio by the auto-derived cap and build the ensemble.

        ``clamp_ratio=False`` is used by LOCO folds, which must reuse the
        already-resolved main ratio verbatim instead of re-capping it against
        the fold's own source size.
        """
        cfg = self.cfg
        auto = round(float(auto_ratio(n_src, cfg.auto_ratio_base,
                                      cfg.auto_ratio_cap)), 4)
        base = round(float(cfg.ratio), 4)
        if requested is None:
            eff = auto
        elif clamp_ratio:
            eff = round(min(float(requested), auto), 4)
        else:
            eff = round(float(requested), 4)
        eff = float(np.clip(eff, 1e-4, 1.0))
        if not do_ensemble:
            return (eff,)
        return tuple(sorted(set([base, eff])))

    @torch.no_grad()
    def _predict_once(self, a_pred_ctrl, a_src_ctrl, a_stim, ratio, cfg):
        """One CAOT pass at a single ratio."""
        dc = cfg.content_dim
        Zp = self.latent(a_pred_ctrl)
        Zs = self.latent(a_src_ctrl)
        Zt = self.latent(a_stim)
        zc_p, zr_p = Zp[:, :dc], Zp[:, dc:]
        zc_s, zr_s = Zs[:, :dc], Zs[:, dc:]
        zc_t, zr_t = Zt[:, :dc], Zt[:, dc:]

        diag = caot_transfer(
            zc_target=zc_p, zc_source=zc_s, zr_source=zr_s,
            zr_stim=zr_t, zc_stim=zc_t,
            ratio=ratio, ot_reg=cfg.ot_reg, source_tau=cfg.source_tau,
            attn_tau=cfg.attn_tau, confidence_floor=cfg.confidence_floor,
            use_pot=cfg.use_pot,
        )

        # Only the response segment changes; content is preserved (Innovation I)
        zr_out = zr_p + diag["gate"][:, None] * diag["delta_pred"]
        z_out = np.concatenate([zc_p, zr_out], axis=1).astype(np.float32)
        X_out = self.decoder(
            torch.as_tensor(z_out, dtype=torch.float32, device=self.device)
        ).cpu().numpy()
        X_out = np.clip(X_out, 0.0, None).astype(np.float32)
        return X_out, zc_p, zr_out, diag

    @torch.no_grad()
    def predict(self, train_adata, cell_to_pred, key_dic=None, ratio=None,
                return_diagnostics=False, run_loco=None, ensemble=None,
                clamp_ratio=True, verbose=True):
        cfg = self.cfg
        kd = cfg.override(key_dic)
        ctk, cdk = kd["cell_type_key"], kd["condition_key"]
        ctrl, stim, predk = kd["ctrl_key"], kd["stim_key"], kd["pred_key"]
        obs = train_adata.obs
        ct = np.asarray(obs[ctk]).astype(str)
        cd = np.asarray(obs[cdk]).astype(str)

        m_pred_ctrl = (ct == str(cell_to_pred)) & (cd == str(ctrl))
        m_src_ctrl = (ct != str(cell_to_pred)) & (cd == str(ctrl))
        # The target type's stimulated cells are the ground truth for this
        # prediction and must never act as a transfer source. The original
        # pipeline relies on the caller withholding them; excluding them here
        # makes that safety guarantee explicit and idempotent.
        m_stim = (cd == str(stim)) & (ct != str(cell_to_pred))
        if not bool(np.any(m_pred_ctrl)):
            raise ValueError(f"no control cells found for {cell_to_pred}")
        if not bool(np.any(m_src_ctrl)) or not bool(np.any(m_stim)):
            raise ValueError("cross-cell-type prediction needs other types' controls and stimulated cells")

        a_pred_ctrl = train_adata[m_pred_ctrl].copy()
        a_src_ctrl = train_adata[m_src_ctrl].copy()
        a_stim = train_adata[m_stim].copy()

        n_src = int(m_src_ctrl.sum())
        do_ensemble = cfg.ensemble_on_predict if ensemble is None else ensemble
        ratios = self._resolve_ratios(ratio, n_src, do_ensemble, clamp_ratio)
        self.last_ratios = ratios
        self._main_ratios = ratios

        outputs, diags, latent_pack = [], [], None
        for r in ratios:
            X_out, zc_p, zr_out, diag = self._predict_once(
                a_pred_ctrl, a_src_ctrl, a_stim, r, cfg)
            outputs.append(X_out)
            diags.append(diag)
            if latent_pack is None:
                latent_pack = (zc_p, zr_out)
            if verbose:
                print(
                    "Predict done | {} cells | top_k={} | mean_gate={:.3f} | "
                    "delta_std={:.4f}".format(
                        a_pred_ctrl.n_obs, diag["top_k"],
                        float(diag["gate"].mean()),
                        float(diag["delta_pred"].std()),
                    )
                )

        X = ensemble_X(outputs)
        zc_p, zr_out = latent_pack
        pred_adata = sc.AnnData(
            X=np.asarray(X, dtype=np.float32),
            obs=a_pred_ctrl.obs.copy(),
            var=a_pred_ctrl.var.copy(),
        )
        # obs condition is often categorical; replace it with an object column
        pred_adata.obs[cdk] = pd.Series(
            [predk] * pred_adata.n_obs, index=pred_adata.obs_names, dtype=object
        )
        pred_adata.obs["scresid_confidence"] = diags[0]["gate"].astype(np.float32)
        pred_adata.obsm["X_scresid_content"] = zc_p.astype(np.float32)
        pred_adata.obsm["X_scresid_response"] = zr_out.astype(np.float32)
        pred_adata.uns["scresid"] = {
            "innovations": "DC-RL + CAOT + LOCO-SC",
            "cell_to_pred": str(cell_to_pred),
            "n_test": int(pred_adata.n_obs),
            "n_src": int(n_src),
            "n_stim": int(a_stim.n_obs),
            "top_k": int(diags[0]["top_k"]),
            "ratio": float(ratios[-1]),
            "ensemble_ratios": tuple(float(r) for r in ratios),
            "ot_reg": float(cfg.ot_reg),
            "content_dim": int(cfg.content_dim),
            "resp_dim": int(cfg.resp_dim),
            "mean_gate": float(diags[0]["gate"].mean()),
            "delta_std": float(diags[0]["delta_pred"].std()),
            "delta_norm_mean": float(np.linalg.norm(diags[0]["delta_pred"], axis=1).mean()),
            "key_dic": dict(kd),
        }

        do_loco = cfg.loco_on_predict if run_loco is None else run_loco
        if do_loco:
            plan = loco_calibrate(
                self, train_adata, cell_to_pred, kd,
                ratio=ratios[-1], ratios=ratios, verbose=verbose,
            )
            pred_adata.uns["scresid"]["calibration"] = plan.to_dict()

        if return_diagnostics:
            return pred_adata, diags
        return pred_adata
