import torch
import torch.nn as nn


class ContactHead(nn.Module):
    def __init__(
        self,
        in_channels=1280,
        reduce_channels=256,     # reduce before flatten
        hidden_dim=2048,         # MLP hidden size
        num_layers=4,            # number of MLP layers
        num_vertices=6890,
        dropout=0.3
    ):
        super().__init__()

        # 1x1 conv to reduce channel dimension (NO pooling)
        self.channel_reduce = nn.Conv2d(
            in_channels,
            reduce_channels,
            kernel_size=1
        )

        flattened_dim = reduce_channels * 32 * 32

        mlp_layers = []
        in_dim = flattened_dim

        for i in range(num_layers - 1):
            mlp_layers.append(nn.Linear(in_dim, hidden_dim))
            mlp_layers.append(nn.GELU())
            mlp_layers.append(nn.LayerNorm(hidden_dim))
            mlp_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # Final layer
        mlp_layers.append(nn.Linear(hidden_dim, num_vertices))

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        """
        x: (B,1280,32,32)
        """
        x = self.channel_reduce(x)   # (B,reduce_channels,32,32)    x (B, 1280, 32, 32) -> (B, 256, 32, 32)
        x = x.flatten(1)             # (B,reduce_channels*32*32)
        x = self.mlp(x)              # (B,6890)

        return torch.sigmoid(x)      # or remove and use BCEWithLogitsLoss