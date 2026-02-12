"""
Contact Prediction Head for SAM-3D-Body

Predicts vertex-wise contact probabilities from SAM-3D-Body
per-vertex features.

Author: you :)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContactPredictionHead(nn.Module):
    """
    Vertex-wise contact prediction head.

    Input:
        vertex_features: (B, V, C)

    Output:
        contact_logits: (B, V, 2)
        contact_probs:  (B, V)
    """

    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ):
        super().__init__()

        layers = []

        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))

        # Binary classification: contact / no-contact
        # layers.append(nn.Linear(hidden_dim, 2))
        layers.append(nn.Linear(hidden_dim, 10475))     # smplx vertices: 10475

        self.mlp = nn.Sequential(*layers)

    def forward(self, vertex_features):
        """
        Args:
            vertex_features (Tensor): (B, V, C)

        Returns:
            dict with:
              - logits: (B, V, 2)
              - probs:  (B, V)
        """
        B, V, C = vertex_features.shape

        logits = self.mlp(vertex_features)  # (B, V, 2)
        probs = F.softmax(logits, dim=-1)[..., 1]  # contact probability

        return {
            "logits": logits,
            "probs": probs
        }
