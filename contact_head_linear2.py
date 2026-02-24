import torch
import torch.nn as nn


class ContactHead(nn.Module):
    def __init__(
        self,
        in_channels=1280,
        reduce_channels=64,      # safe for 12GB
        hidden_dim=1024,
        num_layers=5,
        num_vertices=6890,
        dropout=0.4
    ):
        super().__init__()

        # Reduce channel dimension (NO pooling)
        self.channel_reduce = nn.Conv2d(
            in_channels,
            reduce_channels,
            kernel_size=1
        )

        flattened_dim = reduce_channels * 32 * 32  # 64*32*32 = 65,536

        layers = []
        in_dim = flattened_dim

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, num_vertices))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (1,1280,32,32)
        """
        x = self.channel_reduce(x)   # (1,64,32,32)
        x = x.flatten(1)             # (1,65536)
        x = self.mlp(x)              # (1,6890)

        return x   # use BCEWithLogitsLoss