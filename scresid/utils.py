import scanpy, torch

def adata_process(adata, min_genes=200, min_cells=10, max_value=10, n_top_genes=6000):
    print(f"Filter cells with min_genes={min_genes}...")
    scanpy.pp.filter_cells(adata, min_genes=min_genes)
    print(f"Filter genes with min_cells={min_cells}...")
    scanpy.pp.filter_genes(adata, min_cells=min_cells)
    print("Normalize per cell...")
    scanpy.pp.normalize_per_cell(adata)
    print("Apply log1p to adata...")
    scanpy.pp.log1p(adata)
    print(f"Scale with max_value={max_value}...")
    scanpy.pp.scale(adata, max_value=max_value)
    print(f"Compute highly variable genes with n_top_genes={n_top_genes}...")
    scanpy.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    print("Save highly variable genes to adata...")
    adata = adata[:, adata.var.highly_variable]
    print("Convert adata to DataFrame...")
    df = adata.to_df().clip(lower=0)
    print("Save DataFrame to X of adata....")
    adata.X = df
    print("Preprocess completed.")
    return adata

class AnnDataSet(torch.utils.data.Dataset):
    def __init__(self, adata):
        self.data = adata.to_df().values

    def __getitem__(self, index):
        x = self.data[index, :]
        return x

    def __len__(self):
        return self.data.shape[0]