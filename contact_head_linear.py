import torch
import torch.nn as nn

class ContactHead(nn.Module):
    def __init__(self, num_vertices=6890):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)   # (B,1280,1,1)

        self.mlp = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(2048, num_vertices)
        )

    def forward(self, feat):
        x = self.pool(feat)              # (B,1280,1,1) <- (B,1280,32,32)
        x = x.view(x.size(0), -1)        # (B,1280)
        x = self.mlp(x)                  # (B,6890) contact_logits
        x = torch.sigmoid(x)             # 0~1  contact_prob
        return x