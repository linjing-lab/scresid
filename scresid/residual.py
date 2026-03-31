import torch

class ResBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.LayerNorm(dim), # LayerNorm + ReLU
            torch.nn.ReLU(),
            torch.nn.Linear(dim, dim),
            torch.nn.LayerNorm(dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.block(x) # residual connection

class EfficientEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.in_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.res1 = ResBlock(hidden_dim)
        self.out_proj = torch.nn.Linear(hidden_dim, latent_dim * 2)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.res1(x)
        return self.out_proj(x)

class EfficientDecoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.in_proj = torch.nn.Linear(latent_dim, hidden_dim)
        self.res1 = ResBlock(hidden_dim)
        self.out_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        z = self.in_proj(z)
        z = self.res1(z)
        return self.out_proj(z)