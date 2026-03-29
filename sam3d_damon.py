# models/sam3d_damon.py
import torch.nn as nn
# from sam3d_body.models import build_model
# from models.contact_head import ContactPredictionHead
from sam_3d_body.build_models import load_sam_3d_body
# from contact_head import ContactPredictionHead
# from contact_head_linear import ContactHead
# from contact_head import ContactHead
from contact_head_linear5 import ContactHead
import torch

from util_smpl import smpl_to_uv_batch
import smplx

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

        self.contact_head = ContactHead(in_channels=1280, hidden_dim=256, num_gcn_layers=3)       # TODO: use a simple linear head firstly.

        # self.adjacencyMatrix = torch.load('adjacency.pth')


        # self.smpl_layer = smplx.create(
        #     model_path="./data/models/smpl/SMPL_NEUTRAL.pkl",
        #     model_type="smpl",
        #     gender="neutral",
        #     batch_size=1,  # replace dynamically if needed
        # )

        self.smpl = smplx.create(
            model_path="./data/models/smpl/SMPL_NEUTRAL.pkl",
            model_type="smpl",
            gender="neutral",
            batch_size=1,  # replace dynamically if needed
        ).to("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, batch, label):
        out = self.sam3d(batch)

        # vertex_features = out["vertex_features"]  # (B,V,256)
        # contact = self.contact_head(vertex_features)
        # image_embeddings = out["image_embeddings"]  # (batch_size,1280,32,32)
        # verts_uv = torch.ones(1, 6890, 2)   # TODO:add uv position
        # gt_pose = label["pose"]
        # gt_shape = label["shape"]
        # gt_cam = label["cam"]
        # verts_uv = label["verts_uv"]    # (1,6890,2)
        # pred_pose = out['mhr']['smpl_pose']
        # pred_shape = out['mhr']['smpl_shape']
        # pred_cam = out['mhr']['pred_cam']

        # verts_uv = smpl_to_uv_batch(
        #     pose=gt_pose,  # (B,72)
        #     shape=gt_shape,  # (B,10)
        #     K=gt_cam,  # (B,3,3)
        #     H_img=32,  # H_img
        #     W_img=32,  # W_img
        #     smpl_model_path="./data/smpl/SMPL_NEUTRAL.pkl",
        #     gender="neutral",
        #     device="cuda"
        # )
        # self.adjacencyMatrix = self.adjacencyMatrix.to(image_embeddings.device)
        # contact_probs = self.contact_head(image_embeddings, verts_uv)  # , self.adjacencyMatrix

        mhr = out["mhr"]
        smpl_pose = mhr["smpl_pose"]            # [B,72]
        smpl_shape = mhr["smpl_shape"]          # [B,10]
        pred_cam_t = mhr["pred_cam_t"]          # [B,3]
        cam_int = batch["cam_int"]              # [B,3,3]
        feat_map = out["image_embeddings"]      # [B,1280,Hf,Wf]
        ori_img_size = batch["ori_img_size"]    # [B,2], (W,H)  [B,1,2]
        ori_img_size = ori_img_size.squeeze(1)  # [B,2]

        B, C, Hf, Wf = feat_map.shape


        # 1) verts from SMPL
        smpl_out = self.smpl(
            betas=smpl_shape,
            global_orient=smpl_pose[:, :3],
            body_pose=smpl_pose[:, 3:],  # 69 dims
            return_verts=True,
        )
        verts = smpl_out.vertices  # [B,6890,3]

        # 2) project to UV
        # verts_cam = verts + pred_cam_t[:, None, :]
        # uv_pix = self._project_with_K(verts_cam, cam_int)  # original image pixels

        # grid = normalize_to_minus1_plus1(uv, H, W)  # for grid_sample

        # 3) vertex features
        # vfeat = sample_grid(feat_map, grid)  # [B,6890,C]
        # itok = feat_map.flatten(2).transpose(1, 2)  # [B,H*W,C]

        # 4) cross-attn + classifier
        # vctx = mha(query=vfeat, key=itok, value=itok)
        # logits = head(vctx).squeeze(-1)  # [B,6890]
        # loss = BCEWithLogitsLoss(logits, gt_contact.float())
        # out["contact_probs"] = contact_probs

        # map uv from original image to feature-map coordinates first
        # ori_size = batch["ori_img_size"]  # expected [B,2] => (W,H)
        # sx = Wf / ori_size[:, 0].unsqueeze(1)
        # sy = Hf / ori_size[:, 1].unsqueeze(1)
        # uv_feat = torch.stack([uv_pix[..., 0] * sx, uv_pix[..., 1] * sy], dim=-1)

        # verts_uv_grid = self._uv_to_grid(uv_feat, Hf, Wf)  # [-1,1]
        # contact_logits = self.contact_head(feat_map, verts_uv_grid, adjacency=None)  # [B,6890]
        #
        # out["contact_logits"] = contact_logits
        # out["contact_probs"] = torch.sigmoid(contact_logits)
        # out["pred_smpl_vertices"] = verts

        verts_cam = verts + pred_cam_t[:, None, :]
        uv_px = self._project_with_K(verts_cam, cam_int)  # [B,6890,2]
        verts_uv = self._pixels_to_grid(uv_px, Hf, Wf, ori_img_size)  # [-1,1]

        contact_logits = self.contact_head(feat_map, verts_uv, adjacency=None)

        out["pred_smpl_vertices"] = verts
        out["verts_uv"] = verts_uv
        out["contact_logits"] = contact_logits
        out["contact_probs"] = torch.sigmoid(contact_logits)

        return out


    # helper inside class
    # def _project_with_K(self, points3d, K):
    #     # points3d: [B,N,3], K:[B,3,3]
    #     x, y, z = points3d[...,0], points3d[...,1], points3d[...,2].clamp(min=1e-6)
    #     fx, fy = K[:,0,0].unsqueeze(1), K[:,1,1].unsqueeze(1)
    #     cx, cy = K[:,0,2].unsqueeze(1), K[:,1,2].unsqueeze(1)
    #     u = fx * (x / z) + cx
    #     v = fy * (y / z) + cy
    #     return torch.stack([u, v], dim=-1)  # [B,N,2]
    #
    # def _uv_to_grid(self, uv, H, W):
    #     u = 2.0 * (uv[...,0] / (W - 1)) - 1.0
    #     v = 2.0 * (uv[...,1] / (H - 1)) - 1.0
    #     return torch.stack([u, v], dim=-1)


    def _project_with_K(self, points3d, K):
        x = points3d[..., 0]
        y = points3d[..., 1]
        z = points3d[..., 2].clamp(min=1e-6)
        fx = K[:, 0, 0].unsqueeze(1)
        fy = K[:, 1, 1].unsqueeze(1)
        cx = K[:, 0, 2].unsqueeze(1)
        cy = K[:, 1, 2].unsqueeze(1)
        u = fx * (x / z) + cx
        v = fy * (y / z) + cy
        return torch.stack([u, v], dim=-1)

    def _pixels_to_grid(self, uv_px, H_feat, W_feat, ori_img_size):
        # ori_img_size expected [B,2] as (W,H)
        sx = W_feat / ori_img_size[:, 0].unsqueeze(1).clamp(min=1e-6)
        sy = H_feat / ori_img_size[:, 1].unsqueeze(1).clamp(min=1e-6)
        u_feat = uv_px[..., 0] * sx
        v_feat = uv_px[..., 1] * sy
        # The error is because W_feat/H_feat are Python int (from feat_map.shape), and int has no .clamp(...) method.
        # u = 2.0 * (u_feat / (W_feat - 1).clamp(min=1 if isinstance(W_feat, torch.Tensor) else 1)) - 1.0
        # v = 2.0 * (v_feat / (H_feat - 1).clamp(min=1 if isinstance(H_feat, torch.Tensor) else 1)) - 1.0

        # W_feat/H_feat are ints -> use Python max instead of tensor clamp
        denom_w = max(W_feat - 1, 1)
        denom_h = max(H_feat - 1, 1)

        u = 2.0 * (u_feat / denom_w) - 1.0
        v = 2.0 * (v_feat / denom_h) - 1.0

        return torch.stack([u, v], dim=-1)
