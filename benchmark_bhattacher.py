"""Benchmark driver reproducing the reference log on bhattacher.h5ad.

Run from the directory that contains ``./data/bhattacher.h5ad`` after placing
the ``scresid`` package on PYTHONPATH. The first block is exactly the public
API used by the driving script; the second block adds the LOCO-SC baseline vs
calibrated comparison printed in the reference log.
"""

import warnings

import numpy as np
import scanpy as sc

import scresid

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------- #
# Public API (identical to the driving script)
# --------------------------------------------------------------------- #
adata = sc.read_h5ad("scresid_bundle/data/bhattacher.h5ad")
adata = scresid.adata_process(
    adata,
    # n_top_genes=2000
)
adata.obs.rename(columns={"label": "condition"}, inplace=True)  # bhattacher
# Categorical columns cannot accept new labels via replace; normalize the dtype.
if str(adata.obs["condition"].dtype) == "category":
    adata.obs["condition"] = adata.obs["condition"].astype(str)
adata.obs["condition"].replace({"Maintenance_Cocaine": "control"}, inplace=True)
adata.obs["condition"].replace({"withdraw_15d_Cocaine": "stimulated"}, inplace=True)
adata.obs["condition"].replace({"withdraw_48h_Cocaine": "stimulated"}, inplace=True)
adata = sc.AnnData(adata.X, obs=adata.obs.copy(), var=adata.var.copy())
adata.obs_names_make_unique()
print(adata)
print("\n", adata.obs["condition"].value_counts())
print("\n", adata.obs["cell_type"].value_counts())

model = scresid.SCRESID(input_dim=adata.n_vars, device="cuda:0")
model = model.to(model.device)

key_dic = {
    "condition_key": "condition",
    "cell_type_key": "cell_type",
    "ctrl_key": "control",
    "stim_key": "stimulated",
    "pred_key": "predict",
}

cell_to_pred = "NF Oligo"
# The training set does not contain the target type after perturbation.
train = adata[~((adata.obs[key_dic["cell_type_key"]] == cell_to_pred)
                & (adata.obs[key_dic["condition_key"]] == key_dic["stim_key"]))].copy()
model.train_SCRESID(train, lr=5e-5, epochs=100)

adata_to_pred = adata[((adata.obs[key_dic["cell_type_key"]] == cell_to_pred)
                       & (adata.obs[key_dic["condition_key"]] == key_dic["ctrl_key"]))].copy()
pred = model.predict(
    train_adata=train,
    cell_to_pred=cell_to_pred,
    key_dic=key_dic,
    ratio=0.5,  # clamped by the adaptive ratio cap derived from the source size
)
print(pred)

# --------------------------------------------------------------------- #
# Baseline vs LOCO-SC calibrated evaluation
# --------------------------------------------------------------------- #
pred_cal = scresid.apply_calibration(pred, train, cell_to_pred, key_dic=key_dic)

eval_base = scresid.make_eval_adata(adata, pred, cell_to_pred, key_dic)
eval_cal = scresid.make_eval_adata(adata, pred_cal, cell_to_pred, key_dic)

print("\n===== evaluate : baseline (DC-RL + CAOT + ensemble) =====")
m_base = scresid.evaluate(eval_base, cell_to_pred, key_dic, pred_adata=pred)

print("\n===== evaluate : calibrated (+ response anchor + var alignment) =====")
m_cal = scresid.evaluate(eval_cal, cell_to_pred, key_dic, pred_adata=pred_cal)

print()
scresid.print_comparison(m_base, m_cal)
