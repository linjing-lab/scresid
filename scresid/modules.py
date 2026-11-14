"""Network building blocks: residual encoder / decoder.

The identity-mapping idea of the original ResBlock is kept, with three
engineering adjustments: pre-LayerNorm + SiLU, stackable residual blocks, and
optional dropout. The latent vector is semantically split into
``[content | response]`` in ``model.py``, not here.
"""

from __future__ import annotations

from torch import nn


class ResBlock(nn.Module):
    """LayerNorm -> SiLU -> Linear -> Dropout -> LayerNorm -> SiLU -> Linear + skip."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class EfficientEncoder(nn.Module):
    """Emit ``[mu | logvar]`` with ``2 * latent_dim`` output features."""

    def __init__(self, input_dim, hidden_dim, latent_dim, n_blocks=2, dropout=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [ResBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.out_proj = nn.Linear(hidden_dim, latent_dim * 2)

    def forward(self, x):
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.out_proj(h)


class EfficientDecoder(nn.Module):
    """Decode ``[z_content | z_response]`` back to gene-expression space."""

    def __init__(self, latent_dim, hidden_dim, output_dim, n_blocks=2, dropout=0.0):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [ResBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        h = self.in_proj(z)
        for block in self.blocks:
            h = block(h)
        return self.out_proj(h)
