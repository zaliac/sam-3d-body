import torch
import torch.nn as nn
import torch.nn.functional as F
import smplx


class CrossAttnBlock(nn.Module):
    def __init__(self, dim=1280, heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q_tokens, kv_tokens):
        qn = self.norm_q(q_tokens)
        kvn = self.norm_kv(kv_tokens)
        attn_out, _ = self.attn(qn, kvn, kvn)
        x = q_tokens + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x

# CanonicalSMPL
class ContactHead(nn.Module):
    """
    Canonical SMPL contact head:
      1) Build canonical SMPL verts (pose=0, shape=0): (6890,3)
      2) Create vertex tokens in 1280-dim
      3) Cross-attend with image feat_map tokens
      4) Predict per-vertex contact
    """

    def __init__(
        self,
        smpl_model_dir,
        smpl_gender="neutral",
        num_betas=10,
        d_model=1280,
        num_verts=6890,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_verts = num_verts

        # ---- Build canonical SMPL vertices (once) ----
        smpl = smplx.create(
            model_path=smpl_model_dir,
            model_type="smpl",
            gender=smpl_gender,
            use_pca=False,
            batch_size=1,
            num_betas=num_betas,
        )
        smpl.eval()
        for p in smpl.parameters():
            p.requires_grad = False
        self.smpl = smpl

        with torch.no_grad():
            betas = torch.zeros(1, num_betas)
            body_pose = torch.zeros(1, 23 * 3)  # axis-angle
            global_orient = torch.zeros(1, 3)   # axis-angle
            transl = torch.zeros(1, 3)
            out = self.smpl(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
                return_verts=True,
            )
            canonical_verts = out.vertices[0]  # (6890, 3)
            if canonical_verts.shape[0] != num_verts:
                raise ValueError(
                    f"Expected {num_verts} verts, got {canonical_verts.shape[0]}"
                )

        # buffer: fixed canonical xyz
        self.register_buffer("canonical_verts", canonical_verts, persistent=True)

        # ---- Vertex token construction ----
        # geometric embedding from xyz
        self.vert_xyz_proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.LayerNorm(d_model),
        )
        # # trainable per-vertex identity embedding
        # self.vert_id_emb = nn.Parameter(torch.randn(1, num_verts, d_model) * 0.02)
        # trainable positional embedding
        self.vert_pos_emb = nn.Parameter(torch.randn(1, num_verts, d_model) * 0.02)

        # ---- Cross-attention blocks ----
        self.blocks = nn.ModuleList(
            [CrossAttnBlock(d_model, num_heads, 4.0, dropout) for _ in range(num_layers)]
        )

        # ---- Classifier ----
        # Keep logits for BCEWithLogitsLoss; apply sigmoid only for inference
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def _build_vertex_tokens(self, batch_size):
        # canonical xyz -> 1280
        xyz_feat = self.vert_xyz_proj(self.canonical_verts)  # (6890,1280)
        xyz_feat = xyz_feat.unsqueeze(0).expand(batch_size, -1, -1)  # (B,6890,1280)

        # add trainable id + positional embedding
        # tokens = xyz_feat + self.vert_id_emb + self.vert_pos_emb  # (B,6890,1280)
        tokens = xyz_feat + self.vert_pos_emb  # (B,6890,1280)
        return tokens

    def forward(self, feat_map, return_prob=False):
        """
        feat_map: (B, 1280, 32, 32)
        return:
          logits: (B, 6890)
          prob:   (B, 6890) if return_prob=True
        """
        if feat_map.dim() != 4:
            raise ValueError(f"feat_map must be 4D (B,C,H,W), got {feat_map.shape}")

        B, C, H, W = feat_map.shape
        if C != self.d_model:
            raise ValueError(f"Expected C={self.d_model}, got C={C}")

        # image tokens: (B, 1024, 1280) for 32x32
        img_tokens = feat_map.flatten(2).transpose(1, 2).contiguous()

        # vertex tokens: (B,6890,1280)
        v_tokens = self._build_vertex_tokens(B)

        # cross attention
        for blk in self.blocks:
            v_tokens = blk(v_tokens, img_tokens)

        # per-vertex logits
        logits = self.classifier(v_tokens).squeeze(-1)  # (B,6890)

        if return_prob:
            return logits, torch.sigmoid(logits)
        return logits
