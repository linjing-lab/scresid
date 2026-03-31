# scresid

Single-cell perturbation prediction with residual connections across cell types.

Experiments were executed on NVIDIA A40 of 46068MiB memory in linux with torch==2.1.0+cu121.

## Overview

The scresid is a single-cell gene expression perturbation response prediction model based on Efficient-VDVAE. Its core logic is to use a residual connection module composed of LayerNorm, which is then applied to a variational autoencoder (VAE) to learn the residual loss of low dimensional hidden layers. The high-dimensional sparse gene expression data is mapped to a low dimensional latent space, and the optimal transport technique based on the Sinkhorn algorithm is used to match the cells before and after perturbation, thereby obtaining paired latent representations and corresponding perturbation vectors. Subsequently, the model calculates the similarity between the predicted cells and each cell in the training set through attention mechanism, and weights and fuses their perturbation vectors. Finally, the VAE decoder generates the perturbed gene expression prediction results. This method not only fully considers the heterogeneity at the single-cell level, but also achieves accurate distribution level prediction in scenarios across cell types, significantly better than existing methods.

## install scresid

```python
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install scresid
# pip install numpy==1.26.4
```