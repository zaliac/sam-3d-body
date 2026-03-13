import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Graph Convolution Layer
class GraphConv(nn.Module):
    """
    Simple GCN layer using adjacency matrix
    """

    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x, A):
        """
        x : (B, N, D)
        A : (N, N)
        """

        x = torch.matmul(A, x)   # propagate neighbors
        x = self.lin(x)

        return F.relu(x)

# 2. Vertex ↔ Image Cross Attention
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


# 3
class ContactHead(nn.Module):

    def __init__(
            self,
            in_channels=1280,
            hidden_dim=256,
            num_gcn_layers=3
    ):
        super().__init__()

        # reduce vertex feature
        self.reduce = nn.Linear(in_channels, hidden_dim)

        # cross attention
        self.cross_attn = VertexImageCrossAttention(
            v_dim=hidden_dim,
            img_dim=in_channels
        )

        # GCN layers
        self.gcn_layers = nn.ModuleList([
            GraphConv(hidden_dim)
            for _ in range(num_gcn_layers)
        ])

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    # ------------------------------------------------
    # vertex feature sampling
    # ------------------------------------------------
    def sample_vertex_features(self, feat_map, verts_uv):
        """
        feat_map : (B,C,H,W)
        verts_uv : (B,N,2) in [-1,1]

        return  : (B,N,C)
        """

        grid = verts_uv.unsqueeze(2)

        sampled = F.grid_sample(
            feat_map,
            grid,
            mode="bilinear",
            align_corners=True
        )

        sampled = sampled.squeeze(-1).permute(0, 2, 1)

        return sampled

    # ------------------------------------------------
    # forward
    # ------------------------------------------------
    def forward(self, feat_map, verts_uv, adjacency=None):
        """
        feat_map : (B,1280,32,32)
        verts_uv : (B,6890,2)
        adjacency: (6890,6890)
        """

        B = feat_map.shape[0]

        # ------------------------------------------------
        # 1. vertex feature sampling
        # ------------------------------------------------
        v_feat = self.sample_vertex_features(
            feat_map,
            verts_uv
        )  # (B,6890,1280)

        # ------------------------------------------------
        # 2. feature reduction
        # ------------------------------------------------
        v_feat = self.reduce(v_feat)  # (B,6890,256)

        # ------------------------------------------------
        # 3. image tokens
        # ------------------------------------------------
        img_tokens = feat_map.flatten(2).permute(0, 2, 1)   # (B,1024,1280)
        # (B,1024,1280)

        # ------------------------------------------------
        # 4. cross attention
        # ------------------------------------------------
        v_feat = self.cross_attn(
            v_feat,
            img_tokens
        )  # (B,6890,256)

        # ------------------------------------------------
        # 5. graph convolution
        # ------------------------------------------------
        # adjacency = adjacency.to(v_feat.device)
        if adjacency is not None:
            for gcn in self.gcn_layers:
                v_feat = gcn(v_feat, adjacency)

        # ------------------------------------------------
        # 6. classifier
        # ------------------------------------------------
        logits = self.classifier(v_feat).squeeze(-1)

        prob = torch.sigmoid(logits)

        return prob

    def sample_vertex_features_fast(feat_map, verts_xy):
        B, C, H, W = feat_map.shape
        N = verts_xy.shape[1]

        x = verts_xy[..., 0]
        y = verts_xy[..., 1]

        # 4 neighbor pixels
        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1

        x0 = x0.clamp(0, W - 1) # forces all values in a tensor to stay between min and max
        x1 = x1.clamp(0, W - 1)
        y0 = y0.clamp(0, H - 1)
        y1 = y1.clamp(0, H - 1)

        # bilinear weights
        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        # gather features
        feat = feat_map.permute(0, 2, 3, 1)  # (B,H,W,C)

        Ia = feat[:, y0, x0]
        Ib = feat[:, y1, x0]
        Ic = feat[:, y0, x1]
        Id = feat[:, y1, x1]

        out = (
                wa.unsqueeze(-1) * Ia +
                wb.unsqueeze(-1) * Ib +
                wc.unsqueeze(-1) * Ic +
                wd.unsqueeze(-1) * Id
        )

        return out


# test
def test():
    B = 2

    feat_map = torch.randn(B,1280,32,32)
    verts_uv = torch.randn(B,6890,2)

    adjacency = torch.randn(6890,6890)

    model = ContactHead()

    prob = model(
        feat_map,
        verts_uv,
        adjacency
    )

    print(prob.shape)