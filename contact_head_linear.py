import torch
import torch.nn as nn

class ContactHead(nn.Module):
    def __init__(self, num_vertices=6890):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)   # (B,1280,1,1)

        self.channel_reduce = nn.Conv2d(
            1280,
            64,
            kernel_size=1
        )

        flattened_dim = 64 * 32 * 32

        self.mlp = nn.Sequential(
            nn.Linear(64*32*32, 10240),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(10240, 10240),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(10240, num_vertices)
        )

    def forward(self, x):
        # x = self.pool(feat)              # (B,1280,1,1) <- (B,1280,32,32)
        x = self.channel_reduce(x)          # (B,64,32,32)
        x = x.view(x.size(0), -1)        # (B,1280)
        x = self.mlp(x)                  # (B,6890) contact_logits
        x = torch.sigmoid(x)             # 0~1  contact_prob
        return x