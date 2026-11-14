"""SCRESID: single-cell perturbation response prediction with three innovations.

* ``DC-RL``   - Disentangled Condition-Residual Latent        (model.py)
* ``CAOT``    - Content-guided Attentive Optimal Transport    (transport.py)
* ``LOCO-SC`` - Leave-One-Cell-type-Out Self-Calibration with
                an adaptive ratio ensemble                   (calibration.py)

Public API mirrors the driving script::

    import scresid
    adata = scresid.adata_process(adata)
    model = scresid.SCRESID(input_dim=adata.n_vars, device='cuda:0')
    model = model.to(model.device)
    model.train_SCRESID(train, lr=5e-5, epochs=100)
    pred = model.predict(train_adata=train, cell_to_pred='Endo',
                         key_dic=key_dic, ratio=0.5)

Imports are lazy: importing this package alone does not import torch or
scanpy, so the NumPy-only modules (transport / calibration) remain usable in
lightweight environments and can be numerically self-tested. Torch, scanpy
and tqdm are expected to be pre-installed; this package never downloads or
installs any of them.
"""

from importlib import import_module

__all__ = [
    "SCRESIDConfig", "DEFAULT_KEY_DIC",
    "adata_process", "encode_labels", "PerturbDataset",
    "ResBlock", "EfficientEncoder", "EfficientDecoder",
    "cosine_similarity", "sinkhorn_coupling", "caot_transfer",
    "SCRESID", "evaluate", "draw_reg", "make_eval_adata", "print_comparison",
    "COMPARISON_KEYS",
    "pearson_r2", "metric_r2", "combined_score", "top_deg_index",
    "shared_response", "variance_target", "calibrate_X", "grid_search",
    "aggregate_grid", "auto_ratio", "ensemble_X", "response_support",
    "gate_response_X", "decide_by_loco", "CalibrationPlan",
    "loco_calibrate", "apply_calibration", "support_gate_calibration",
    "obs_arrays", "matrix",
]

_LAZY = {
    "SCRESIDConfig": ("config", "SCRESIDConfig"),
    "DEFAULT_KEY_DIC": ("config", "DEFAULT_KEY_DIC"),
    "adata_process": ("data", "adata_process"),
    "encode_labels": ("data", "encode_labels"),
    "PerturbDataset": ("data", "PerturbDataset"),
    "ResBlock": ("modules", "ResBlock"),
    "EfficientEncoder": ("modules", "EfficientEncoder"),
    "EfficientDecoder": ("modules", "EfficientDecoder"),
    "cosine_similarity": ("transport", "cosine_similarity"),
    "sinkhorn_coupling": ("transport", "sinkhorn_coupling"),
    "caot_transfer": ("transport", "caot_transfer"),
    "SCRESID": ("model", "SCRESID"),
    # The evaluation helpers live in metrics.py, NOT in a submodule called
    # evaluate.py: a submodule named ``evaluate`` would shadow the exported
    # ``scresid.evaluate`` function as soon as any sibling export is resolved.
    "evaluate": ("metrics", "evaluate"),
    "draw_reg": ("metrics", "draw_reg"),
    "make_eval_adata": ("metrics", "make_eval_adata"),
    "print_comparison": ("compare", "print_comparison"),
    "COMPARISON_KEYS": ("compare", "COMPARISON_KEYS"),
    "pearson_r2": ("calibration", "pearson_r2"),
    "metric_r2": ("calibration", "metric_r2"),
    "combined_score": ("calibration", "combined_score"),
    "top_deg_index": ("calibration", "top_deg_index"),
    "shared_response": ("calibration", "shared_response"),
    "variance_target": ("calibration", "variance_target"),
    "calibrate_X": ("calibration", "calibrate_X"),
    "grid_search": ("calibration", "grid_search"),
    "aggregate_grid": ("calibration", "aggregate_grid"),
    "auto_ratio": ("calibration", "auto_ratio"),
    "ensemble_X": ("calibration", "ensemble_X"),
    "response_support": ("calibration", "response_support"),
    "gate_response_X": ("calibration", "gate_response_X"),
    "decide_by_loco": ("calibration", "decide_by_loco"),
    "CalibrationPlan": ("calibration", "CalibrationPlan"),
    "loco_calibrate": ("calibration", "loco_calibrate"),
    "apply_calibration": ("calibration", "apply_calibration"),
    "support_gate_calibration": ("calibration", "support_gate_calibration"),
    "obs_arrays": ("calibration", "obs_arrays"),
    "matrix": ("calibration", "matrix"),
}


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _LAZY[name]
    module = import_module("." + module_name, __name__)
    value = getattr(module, attr)

    # Defensive: importing a submodule binds ``package.<submodule>`` to the
    # module object. If a future export ever shares that name, restore the
    # callable so ``scresid.<name>(...)`` can never resolve to a module.
    if module_name in _LAZY and module_name in globals():
        globals()[module_name] = getattr(module, _LAZY[module_name][1])

    globals()[name] = value  # cache; later accesses skip the import
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
