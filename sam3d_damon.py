# models/sam3d_damon.py
import torch.nn as nn
# from sam3d_body.models import build_model
# from models.contact_head import ContactPredictionHead
from sam_3d_body.build_models import load_sam_3d_body
from contact_head import ContactPredictionHead

class Sam3DWithContact(nn.Module):
    def __init__(self, checkpoint_path):
    # def __init__(self):
        super().__init__()

        # self.sam3d = build_model(checkpoint_path)
        self.sam3d,self.model_cfg = load_sam_3d_body(checkpoint_path)     # "sam3d_body.pth"     SAM3DBody

        # Freeze image encoder (recommended)
        # for p in self.sam3d.image_encoder.parameters():
        for p in self.sam3d.backbone.parameters():
            p.requires_grad = False

        self.contact_head = ContactPredictionHead(
            in_dim=256,      # SAM-3D-Body vertex feature dim
            hidden_dim=128
        )

    def forward(self, images):
        out = self.sam3d(images)

        vertex_features = out["vertex_features"]  # (B,V,256)
        contact = self.contact_head(vertex_features)

        return {
            "vertices": out["vertices"],
            "contact_logits": contact["logits"],
            "contact_probs": contact["probs"]
        }
