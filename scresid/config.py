"""Centralized configuration for the SCRESID framework.

The framework is organized around three innovations:

* ``DC-RL``   - Disentangled Condition-Residual Latent space      (model.py)
* ``CAOT``    - Content-guided Attentive Optimal Transport        (transport.py)
* ``LOCO-SC`` - Leave-One-Cell-type-Out Self-Calibration with an
                adaptive ratio ensemble                          (calibration.py)

Every hyper-parameter and every default ``obs`` column name lives here, so an
experiment can be reproduced from one config snapshot and checked against the
diagnostics written into ``pred.uns['scresid']``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Sequence

DEFAULT_KEY_DIC: Dict[str, str] = {
    "cell_type_key": "cell_type",
    "condition_key": "condition",
    "ctrl_key": "control",
    "stim_key": "stimulated",
    "pred_key": "predict",
}


@dataclass
class SCRESIDConfig:
    # ---- network / DC-RL structure ----
    input_dim: int = 6000
    latent_dim: int = 100
    content_dim: int = 60          # z = [z_content | z_response]
    hidden_dim: int = 1000
    n_res_blocks: int = 2
    dropout: float = 0.05
    noise_rate: float = 0.1        # denoising input noise, training only

    # ---- DC-RL loss weights ----
    kl_weight: float = 5e-4
    lambda_content: float = 1.0    # identity / condition disentanglement
    lambda_resp: float = 1.0       # response residual constraint
    lambda_decorr: float = 0.1     # content-response decorrelation
    content_margin: float = 1.0

    # ---- CAOT transfer ----
    ot_reg: float = 0.05           # entropic regularization (cost normalized to [0, 1])
    source_tau: float = 0.25       # temperature of the target-aware source marginal
    attn_tau: float = 0.10         # temperature of the cell-level attention
    ratio: float = 0.05            # baseline top-k fraction of source controls
    confidence_floor: float = 0.5  # gate floor, protects OOD target cells
    use_pot: bool = True           # use POT.sinkhorn when available, else built-in Sinkhorn

    # ---- training ----
    lr: float = 5e-5
    weight_decay: float = 1e-5
    grad_clip: float = 10.0

    # ---- LOCO-SC: adaptive ratio ensemble ----
    auto_ratio_base: float = 0.5
    auto_ratio_cap: int = 256      # never activate more than 256 source controls
    ensemble_on_predict: bool = True
    loco_on_predict: bool = True
    loco_min_cells: int = 500      # skip rare cell types in LOCO validation
    loco_top_n: int = 100
    alphas: Sequence[float] = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9)
    var_strengths: Sequence[float] = (0.0, 0.3, 0.6, 1.0)
    loco_w_top: float = 0.5
    loco_w_var: float = 0.5

    key_dic: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_KEY_DIC))

    def __post_init__(self):
        if not 0 < self.content_dim < self.latent_dim:
            raise ValueError("content_dim must lie in (0, latent_dim)")
        if self.ratio <= 0 or self.ratio > 1:
            raise ValueError("ratio must lie in (0, 1]")
        merged = dict(DEFAULT_KEY_DIC)
        merged.update(self.key_dic or {})
        self.key_dic = merged
        for k in DEFAULT_KEY_DIC:
            self.key_dic[k] = str(self.key_dic[k])

    @property
    def resp_dim(self) -> int:
        return self.latent_dim - self.content_dim

    def to_dict(self) -> dict:
        return asdict(self)

    def override(self, key_dic: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Merge a call-site ``key_dic`` without mutating this config."""
        if not key_dic:
            return dict(self.key_dic)
        out = dict(self.key_dic)
        out.update({str(k): str(v) for k, v in key_dic.items()})
        return out
