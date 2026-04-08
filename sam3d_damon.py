# models/sam3d_damon.py
import torch.nn as nn
# from sam3d_body.models import build_model
# from models.contact_head import ContactPredictionHead
from sam_3d_body.build_models import load_sam_3d_body
# from contact_head import ContactPredictionHead
# from contact_head_linear import ContactHead
# from contact_head import ContactHead
from contact_head_linear7 import ContactHead
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

        # self.contact_head = ContactHead(in_channels=1280, hidden_dim=256, num_gcn_layers=3)       # TODO: use a simple linear head firstly.

        self.contact_head = ContactHead(
            smpl_model_dir="./data/models/smpl/SMPL_NEUTRAL.pkl",
            d_model=1280,
            num_verts=6890,
        )
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
        # smpl_out = self.smpl(
        #     betas=smpl_shape,
        #     global_orient=smpl_pose[:, :3],
        #     body_pose=smpl_pose[:, 3:],  # 69 dims
        #     return_verts=True,
        # )
        # verts = smpl_out.vertices  # [B,6890,3]

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

        # verts_cam = verts + pred_cam_t[:, None, :]
        # uv_px = self._project_with_K(verts_cam, cam_int)  # [B,6890,2]
        # verts_uv = self._pixels_to_grid(uv_px, Hf, Wf, ori_img_size)  # [-1,1]
        verts_uv, valid_mask, verts = self.get_gt_verts_uv(label, self.smpl, Hf, Wf, ori_img_size, ori_img_size.device )

        # contact_logits = self.contact_head(feat_map, verts_uv, adjacency=None)
        contact_logits = self.contact_head(feat_map)

        out["pred_smpl_vertices"] = verts
        out["verts_uv"] = verts_uv
        out["contact_logits"] = contact_logits
        out["contact_probs"] = torch.sigmoid(contact_logits)
        out["valid_mask"] = valid_mask

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



    def get_gt_verts_uv(self, label, smpl_layer, feat_h, feat_w, ori_img_size, device):
        """
        label keys:
          - pose: [B,72] or [72]
          - shape: [B,10] or [10]
          - cam_k: [B,3,3] or [3,3]
          - SMPL_root_translation: [B,3] or [3]
        ori_img_size: [B,2] (W,H) in original image pixels
        returns:
          verts_uv_grid: [B,6890,2] in [-1,1] for grid_sample
          valid_mask: [B,6890] (True if inside feature map grid)
        """
        pose = torch.as_tensor(label["pose"], dtype=torch.float32, device=device)
        shape = torch.as_tensor(label["shape"], dtype=torch.float32, device=device)
        K = torch.as_tensor(label["cam_k"], dtype=torch.float32, device=device)
        transl = torch.as_tensor(label["SMPL_root_translation"], dtype=torch.float32, device=device)

        if pose.ndim == 1:
            pose = pose.unsqueeze(0)
        if shape.ndim == 1:
            shape = shape.unsqueeze(0)
        if K.ndim == 2:
            K = K.unsqueeze(0)
        if transl.ndim == 1:
            transl = transl.unsqueeze(0)

        # 1) GT SMPL vertices in body frame
        smpl_out = smpl_layer(
            betas=shape,                    # [B,10]
            global_orient=pose[:, :3],      # [B,3]
            body_pose=pose[:, 3:],          # [B,69]
            return_verts=True,
        )
        verts = smpl_out.vertices           # [B,6890,3]

        # 2) Apply GT root translation (camera/world depending on dataset convention)
        verts_cam = verts + transl[:, None, :]   # [B,6890,3]

        # 3) Perspective projection with GT intrinsics
        x = verts_cam[..., 0]
        y = verts_cam[..., 1]
        z = verts_cam[..., 2].clamp(min=1e-6)

        fx = K[:, 0, 0].unsqueeze(1)
        fy = K[:, 1, 1].unsqueeze(1)
        cx = K[:, 0, 2].unsqueeze(1)
        cy = K[:, 1, 2].unsqueeze(1)

        u_px = fx * (x / z) + cx
        v_px = fy * (y / z) + cy
        # If your dataset uses opposite Y camera convention, try:
        # v_px = fy * (-y / z) + cy

        # 4) Original pixel -> feature-map pixel
        # ori_img_size is (W,H)
        if ori_img_size.ndim == 3:   # e.g., [B,1,2]
            ori_img_size = ori_img_size.squeeze(1)
        ori_w = ori_img_size[:, 0].unsqueeze(1).to(device).clamp(min=1e-6)
        ori_h = ori_img_size[:, 1].unsqueeze(1).to(device).clamp(min=1e-6)

        u_feat = u_px * (feat_w / ori_w)
        v_feat = v_px * (feat_h / ori_h)

        # 5) Feature pixel -> normalized grid [-1,1] (align_corners=True)
        denom_w = max(feat_w - 1, 1)
        denom_h = max(feat_h - 1, 1)

        u = 2.0 * (u_feat / denom_w) - 1.0
        v = 2.0 * (v_feat / denom_h) - 1.0
        verts_uv_grid = torch.stack([u, v], dim=-1)   # [B,6890,2]

        valid_mask = (
            (verts_uv_grid[..., 0] >= -1.0) & (verts_uv_grid[..., 0] <= 1.0) &
            (verts_uv_grid[..., 1] >= -1.0) & (verts_uv_grid[..., 1] <= 1.0)
        )
        return verts_uv_grid, valid_mask, verts     # (1,6890,2), (1,6890), (1,6890,3)
