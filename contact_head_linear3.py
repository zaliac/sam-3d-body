import torch
import torch.nn as nn
import torch.nn.functional as F


class ContactHead(nn.Module):
    """
    Plug-in contact head for SAM-3D body backbone.

    Input:
        feat_map: (B, 1280, 32, 32)
        verts_uv: (B, 6890, 2)   normalized to [-1,1]
        adjacency: (6890, 6890) optional sparse/dense adjacency matrix

    Output:
        contact_prob: (B, 6890)
    """

    def __init__(
        self,
        in_channels=1280,
        hidden_dim=256,
        use_graph_smoothing=False
    ):
        super().__init__()

        self.use_graph_smoothing = use_graph_smoothing

        #
        self.reduce = nn.Linear(in_channels, hidden_dim)

        # MLP head
        self.classifier = nn.Sequential(
            # nn.LayerNorm(hidden_dim),      # nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def sample_vertex_features(self, feat_map, verts_uv):
        """
        feat_map: (B, C, H, W)
        verts_uv: (B, N, 2) in [-1,1]
        return: (B, N, C)
        """
        B, C, H, W = feat_map.shape
        N = verts_uv.shape[1]

        # grid_sample need (B, N, 1, 2)
        grid = verts_uv.unsqueeze(2)        # (1,6890,2) -> (1,6890,1,2): (B, H_out, W_out, 2)

        sampled = F.grid_sample(
            feat_map,
            grid,
            mode="bilinear",
            align_corners=True
        )  # (B, C, N, 1)   (1, 1280, 6890, 1) bilinear interpolation: 在每个顶点投影位置采样 feature map. input: (B,1280,32,32), output: (B,1280,6890,1)

        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # (B, N, C) (1,6890,1280) vertex features.
        return sampled

    def graph_smoothing(self, v_feat, adjacency):
        """
        v_feat: (B, N, D)
        adjacency: (N, N)
        """
        # normailize
        A = adjacency
        deg = A.sum(dim=1, keepdim=True) + 1e-6
        A_norm = A / deg

        # matrix
        v_feat = torch.matmul(A_norm, v_feat)
        return v_feat       # a simple implementation of Graph Convolution

    def forward(self, feat_map, verts_uv, adjacency=None):

        # 1
        v_feat = self.sample_vertex_features(feat_map, verts_uv)    # (1,6890,1280)
        # (B, 6890, 1280)

        # 2
        v_feat = self.reduce(v_feat)    # (1,6890,1280) -> (1,6890,256)
        # (B, 6890, hidden_dim)

        # 3
        if self.use_graph_smoothing and adjacency is not None:
            v_feat = self.graph_smoothing(v_feat, adjacency)

        # 4
        logits = self.classifier(v_feat).squeeze(-1)    # (1, 6890)
        prob = torch.sigmoid(logits)    # 0 ~ 1

        return prob