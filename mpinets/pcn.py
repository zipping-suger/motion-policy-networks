import torch
from torch import nn
import pytorch_lightning as pl


class PCNEncoder(pl.LightningModule):
    def __init__(self, pc_latent_dim):
        super().__init__()

        self._build_model(pc_latent_dim)

    def _build_model(self, pc_latent_dim):
        # First shared MLP (now from 4D input)
        self.first_conv = nn.Sequential(
            nn.Conv1d(4, 128, 1),          # change input channels to 4
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # Second shared MLP
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        
        # Fully connected layers for global feature processing
        self.proj = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GroupNorm(16, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.GroupNorm(16, 2048),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2048, 2048),
        )
        
    def forward(self, xyzf: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyzf: (B, N, 4) — input point cloud (x, y, z, feature)
        
        Returns:
            global_feature: (B, latent_dim)
        """
        B, N, _ = xyzf.shape

        # (B, N, 4) → (B, 4, N)
        x = xyzf.transpose(2, 1).contiguous()

        # First shared MLP
        point_feat = self.first_conv(x)                       # (B, 256, N)

        # Global feature (max pooling over N)
        global_feat = torch.max(point_feat, dim=2, keepdim=True)[0]  # (B, 256, 1)

        # Concatenate global to each point
        expanded_global = global_feat.expand(-1, -1, N)       # (B, 256, N)
        concat_feat = torch.cat([point_feat, expanded_global], dim=1)  # (B, 512, N)

        # Second shared MLP
        refined_feat = self.second_conv(concat_feat)           # (B, latent_dim, N)

        # Final global feature (max pooling)
        global_feature = torch.max(refined_feat, dim=2)[0]     # (B, latent_dim)
        
        feature = self.proj(global_feature)                # (B, 2048)

        return feature


# Example: test run
if __name__ == "__main__":
    B, N = 8, 2048
    xyzf = torch.rand(B, N, 4)
    encoder = PCNEncoder()
    out = encoder(xyzf)
    print(f"Output shape: {out.shape}")  # Should be (B, latent_dim)
