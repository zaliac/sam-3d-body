# models/sam3d_damon.py
import torch.nn as nn
# from sam3d_body.models import build_model
# from models.contact_head import ContactPredictionHead
from sam_3d_body.build_models import load_sam_3d_body
# from contact_head import ContactPredictionHead
# from contact_head_linear import ContactHead
from contact_head_linear3 import ContactHead
import torch

from util_smpl import smpl_to_uv_batch

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

    def forward(self, batch, label):
        out = self.sam3d(batch)

        # vertex_features = out["vertex_features"]  # (B,V,256)
        # contact = self.contact_head(vertex_features)
        image_embeddings = out["image_embeddings"]      # (batch_size,1280,32,32)
        verts_uv = torch.ones(1, 6890, 2)   # TODO:add uv position
        gt_pose = label["pose"]
        gt_shape = label["shape"]
        gt_cam = label["cam"]

        verts_uv = smpl_to_uv_batch(
            pose=gt_pose,  # (B,72)
            shape=gt_shape,  # (B,10)
            K=gt_cam,  # (B,3,3)
            H_img=32,        # H_img
            W_img=32,         # W_img
            smpl_model_path="./data/smpl/SMPL_NEUTRAL.pkl",
            gender="neutral",
            device="cuda"
        )
        contact_probs = self.contact_head(image_embeddings,verts_uv)

        out["contact_probs"] = contact_probs
        # TODO:
        # out_mhr = out["mhr"]
        return out
