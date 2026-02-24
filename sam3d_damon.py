# models/sam3d_damon.py
import torch.nn as nn
# from sam3d_body.models import build_model
# from models.contact_head import ContactPredictionHead
from sam_3d_body.build_models import load_sam_3d_body
# from contact_head import ContactPredictionHead
from contact_head_linear2 import ContactHead

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

        self.contact_head = ContactHead()       # TODO: use a simple linear head firstly.

    def forward(self, batch):
        out = self.sam3d(batch)

        # vertex_features = out["vertex_features"]  # (B,V,256)
        # contact = self.contact_head(vertex_features)
        image_embeddings = out["image_embeddings"]      # (batch_size,1280,32,32)
        contact_probs = self.contact_head(image_embeddings)

        out["contact_probs"] = contact_probs
        # TODO:
        # out_mhr = out["mhr"]
        return out
