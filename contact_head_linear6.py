import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x, A):
        """
        x: (B, N, D)
        A: (N, N)
        """
        x = torch.einsum("ij,bjd->bid", A, x)
        x = self.lin(x)
        return F.relu(x)


class VertexImageCrossAttention(nn.Module):
    def __init__(self, v_dim=256, img_dim=1280, heads=8):
        super().__init__()
        self.q_proj = nn.Linear(v_dim, v_dim)
        self.k_proj = nn.Linear(img_dim, v_dim)
        self.v_proj = nn.Linear(img_dim, v_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=v_dim,
            num_heads=heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(v_dim)

    def forward(self, v_feat, img_tokens):
        q = self.q_proj(v_feat)
        k = self.k_proj(img_tokens)
        v = self.v_proj(img_tokens)

        out, _ = self.attn(q, k, v)
        return self.norm(v_feat + out)


class VertexXYZEncoder(nn.Module):
    def __init__(self, out_dim=256, num_verts=6890, use_vertex_id=True):
        super().__init__()
        self.use_vertex_id = use_vertex_id

        self.mlp = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

        if use_vertex_id:
            self.vertex_id_embed = nn.Embedding(num_verts, out_dim)

    def forward(self, verts_xyz):
        """
        verts_xyz: (B, 6890, 3)
        """
        v_feat = self.mlp(verts_xyz)

        if self.use_vertex_id:
            B, N, _ = verts_xyz.shape
            vid = torch.arange(N, device=verts_xyz.device).unsqueeze(0).expand(B, -1)
            v_feat = v_feat + self.vertex_id_embed(vid)

        return self.norm(v_feat)


class ContactHead(nn.Module):
    def __init__(
        self,
        in_channels=1280,
        hidden_dim=256,
        num_gcn_layers=3,
        num_verts=6890,
    ):
        super().__init__()

        # vertex XYZ -> 256
        self.vertex_encoder = VertexXYZEncoder(
            out_dim=hidden_dim,
            num_verts=num_verts,
            use_vertex_id=True,
        )

        # optional sampled image feature -> 256
        self.reduce_img_feat = nn.Linear(in_channels, hidden_dim)

        # cross attention: vertex query attends to image tokens
        self.cross_attn = VertexImageCrossAttention(
            v_dim=hidden_dim,
            img_dim=in_channels,
            heads=8
        )

        self.gcn_layers = nn.ModuleList([
            GraphConv(hidden_dim) for _ in range(num_gcn_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, 1)
        )

    def sample_vertex_features(self, feat_map, verts_uv):
        """
        feat_map: (B, C, H, W)
        verts_uv: (B, N, 2) in [-1, 1]
        return: (B, N, C)
        """
        grid = verts_uv.unsqueeze(2)  # (B, N, 1, 2)
        sampled = F.grid_sample(
            feat_map,
            grid,
            mode="bilinear",
            align_corners=True
        )
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # (B, N, C)
        return sampled

    def forward(self, feat_map, verts_xyz, verts_uv=None, adjacency=None):
        """
        feat_map: (B, 1280, H, W)
        verts_xyz: (B, 6890, 3)
        verts_uv:  (B, 6890, 2), optional
        adjacency: (6890, 6890), optional
        """

        # 1) embed SMPL vertices from 3D -> 256
        v_feat = self.vertex_encoder(verts_xyz)   # (B, 6890, 256)

        # 2) optionally fuse sampled image feature at each vertex UV
        if verts_uv is not None:
            img_vfeat = self.sample_vertex_features(feat_map, verts_uv)   # (B, 6890, 1280)
            img_vfeat = self.reduce_img_feat(img_vfeat)                   # (B, 6890, 256)
            v_feat = v_feat + img_vfeat

        # 3) image tokens for cross-attention
        img_tokens = feat_map.flatten(2).permute(0, 2, 1)   # (B, H*W, 1280)

        # 4) cross attention
        v_feat = self.cross_attn(v_feat, img_tokens)        # (B, 6890, 256)

        # 5) GCN
        if adjacency is not None:
            for gcn in self.gcn_layers:
                v_feat = gcn(v_feat, adjacency)

        # 6) per-vertex binary contact logit
        logits = self.classifier(v_feat).squeeze(-1)        # (B, 6890)

        return logits
