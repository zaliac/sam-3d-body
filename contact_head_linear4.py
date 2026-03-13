import torch
import torch.nn as nn
import torch.nn.functional as F


class ContactHead(nn.Module):

    def __init__(
        self,
        in_channels=1280,
        hidden_dim=256,
        num_vertices=6890,
        num_transformer_layers=2,
        num_heads=8
    ):
        super().__init__()

        self.num_vertices = num_vertices

        # feature reduction
        self.reduce = nn.Linear(in_channels, hidden_dim)

        # vertex tokens
        self.vertex_embed = nn.Parameter(
            torch.randn(num_vertices, hidden_dim)
        )

        # transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )

        # graph smoothing (optional)
        self.graph_fc = nn.Linear(hidden_dim, hidden_dim)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )


    def sample_vertex_features(self, feat_map, verts_uv):

        B,C,H,W = feat_map.shape
        N = verts_uv.shape[1]

        grid = verts_uv.unsqueeze(2)

        sampled = F.grid_sample(
            feat_map,
            grid,
            mode="bilinear",
            align_corners=True
        )

        sampled = sampled.squeeze(-1).permute(0,2,1)

        return sampled


    def graph_smoothing(self, v_feat, adjacency):

        A = adjacency
        deg = A.sum(1,keepdim=True) + 1e-6
        A_norm = A / deg

        return torch.matmul(A_norm, v_feat)


    def forward(self, feat_map, verts_uv, adjacency=None):

        B = feat_map.shape[0]

        # 1 sample vertex feature
        v_feat = self.sample_vertex_features(feat_map, verts_uv)
        # (B,6890,1280)

        # 2 reduce dim
        v_feat = self.reduce(v_feat)
        # (B,6890,256)

        # 3 add vertex token embedding
        v_feat = v_feat + self.vertex_embed.unsqueeze(0)

        # 4 transformer reasoning
        v_feat = self.transformer(v_feat)

        # 5 graph smoothing
        if adjacency is not None:
            v_feat = self.graph_smoothing(v_feat, adjacency)

        # 6 classification
        logits = self.classifier(v_feat).squeeze(-1)

        prob = torch.sigmoid(logits)

        return prob