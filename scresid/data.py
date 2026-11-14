"""Data layer: preprocessing, label encoding, and the torch Dataset.

Engineering notes
-----------------
* ``scanpy.pp.normalize_total`` replaces the removed ``normalize_per_cell``.
* HVG selection happens after normalize + log1p and before scaling, so the
  variance used for ranking is not flattened by ``pp.scale``.
* ``adata.X`` is always written back as a ``float32`` ndarray.
* The Dataset also returns cell-type and condition labels, which the DC-RL
  disentanglement loss consumes.

Torch and scanpy are expected to be pre-installed in the runtime; this module
never installs or downloads anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import Dataset


def adata_process(
    adata,
    min_genes: int = 200,
    min_cells: int = 10,
    max_value: float = 10,
    n_top_genes: int = 6000,
    target_sum: float = 1e4,
):
    """QC -> library normalization -> log1p -> HVG -> scale -> clip."""
    print(f"Filter cells with min_genes={min_genes} ...")
    sc.pp.filter_cells(adata, min_genes=min_genes)
    print(f"Filter genes with min_cells={min_cells} ...")
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print("Normalize total counts per cell ...")
    sc.pp.normalize_total(adata, target_sum=target_sum)
    print("Apply log1p ...")
    sc.pp.log1p(adata)
    print(f"Compute highly variable genes with n_top_genes={n_top_genes} ...")
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var.highly_variable].copy()
    print(f"Scale with max_value={max_value} ...")
    sc.pp.scale(adata, max_value=max_value)
    X = np.asarray(adata.to_df().clip(lower=0).values, dtype=np.float32)
    adata.X = X
    print(f"Preprocess completed: {adata.n_obs} cells x {adata.n_vars} genes.")
    return adata


def encode_labels(adata, key_dic):
    """Encode cell types and the stimulated flag from ``adata.obs``.

    Returns
    -------
    codes : (n_obs,) int64 cell-type indices
    is_stim : (n_obs,) float32, 1 for stimulated and 0 otherwise
    """
    cell_types = adata.obs[key_dic["cell_type_key"]].astype(str)
    codes, _ = pd.factorize(cell_types, sort=True)
    cond = adata.obs[key_dic["condition_key"]].astype(str).to_numpy()
    is_stim = (cond == str(key_dic["stim_key"])).astype(np.float32)
    return codes.astype(np.int64), is_stim


class PerturbDataset(Dataset):
    """Yield ``{x, cell_type, is_stim}``; labels degrade to zeros if omitted."""

    def __init__(self, adata, cell_type_codes=None, is_stim=None):
        self.x = np.asarray(adata.to_df().values, dtype=np.float32)
        n = self.x.shape[0]
        if cell_type_codes is None:
            cell_type_codes = np.zeros(n, dtype=np.int64)
        if is_stim is None:
            is_stim = np.zeros(n, dtype=np.float32)
        self.cell_type = np.asarray(cell_type_codes, dtype=np.int64).reshape(-1)
        self.is_stim = np.asarray(is_stim, dtype=np.float32).reshape(-1)
        if not (len(self.cell_type) == len(self.is_stim) == n):
            raise ValueError("label length does not match the number of cells")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return {
            "x": torch.from_numpy(self.x[index]),
            "cell_type": int(self.cell_type[index]),
            "is_stim": float(self.is_stim[index]),
        }
