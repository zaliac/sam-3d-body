import torch
import smplx
import numpy as np
import argparse
import os
import sys
from typing import Any, List
from util_smpl import smpl_to_uv_batch

def load_npy(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path, allow_pickle=True)


class SMPL:
    def __init__(self, npz_path: str):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"SMPL npz file not found: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)

        self.v_template = torch.tensor(data['v_template'], dtype=torch.float32)
        self.shapedirs = torch.tensor(data['shapedirs'], dtype=torch.float32)
        self.posedirs = torch.tensor(data['posedirs'], dtype=torch.float32)
        self.J_regressor = torch.tensor(data['J_regressor'], dtype=torch.float32)  # <-- fixed
        self.parents = data['kintree_table'][0].astype(np.int64)
        self.faces = data['f'].astype(np.int64)
        self.num_betas = self.shapedirs.shape[-1]

    def forward(self, betas: torch.Tensor, pose: torch.Tensor):
        """
        Args:
            betas: (10,) shape coefficients
            pose: (72,) axis-angle pose (global_orient + body_pose)
        Returns:
            vertices: (6890,3) tensor
        """
        # Shape blend
        v_shaped = self.v_template + torch.einsum('ijk,k->ij', self.shapedirs, betas)

        # TODO: for simplicity, skip pose blend shapes (or implement if needed)
        # Here we just return v_shaped as vertices
        return v_shaped

def smpl_to_uv_single(pose: np.ndarray, shape: np.ndarray, cam: np.ndarray, H_img: int=32, W_img: int=32) -> np.ndarray:
    """
    用 SMPL + 相机内参生成 verts_uv
    Args:
        pose: (72,)
        shape: (10,)
        cam: (3,3)
        H_img, W_img: 图像尺寸
        smpl_model: 已加载的 SMPL 模型
    Returns:
        verts_uv: (6890,2) ∈ [-1,1]
    """
    pose_t = torch.tensor(pose, dtype=torch.float32).unsqueeze(0)   # (1,72)
    shape_t = torch.tensor(shape, dtype=torch.float32).unsqueeze(0) # (1,10)
    cam_t = torch.tensor(cam, dtype=torch.float32).unsqueeze(0)     # (1,3,3)

    # output = smpl_model(
    #     betas=shape_t,
    #     body_pose=pose_t[:,3:],
    #     global_orient=pose_t[:,:3],
    #     return_verts=True
    # )

    smpl_model = SMPL("./data/models/smpl/SMPL_NEUTRAL.npz")

    # inside loop
    verts_3d = smpl_model.forward(
        betas=torch.tensor(shape_t, dtype=torch.float32),
        pose=torch.tensor(pose_t, dtype=torch.float32)
    )
    # verts_3d = output.vertices  # (1,6890,3)

    X = verts_3d[...,0]
    Y = verts_3d[...,1]
    Z = verts_3d[...,2].clamp(min=1e-6)

    fx = cam_t[:,0,0].unsqueeze(1)
    fy = cam_t[:,1,1].unsqueeze(1)
    cx = cam_t[:,0,2].unsqueeze(1)
    cy = cam_t[:,1,2].unsqueeze(1)

    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    uv_pixels = torch.stack([u,v], dim=-1)  # (1,6890,2)

    # normalize [-1,1]
    u_norm = 2 * (uv_pixels[...,0] / (W_img - 1)) - 1
    v_norm = 2 * (uv_pixels[...,1] / (H_img - 1)) - 1
    verts_uv = torch.stack([u_norm, v_norm], dim=-1).squeeze(0) # (6890,2)

    return verts_uv.cpu().numpy()


def build_samples(imgnames: np.ndarray, labels: np.ndarray, poses: np.ndarray, shapes: np.ndarray, cams: np.ndarray) -> List[dict]:
    """
    构建 sample，同时生成 verts_uv
    """
    B = len(imgnames)
    samples = []
    # smpl_model_path = "./data/models"

    # smpl_model = smplx.create(
    #     model_path="./data/models",  # 传 models 目录
    #     model_type="smpl",
    #     gender="neutral",
    #     ext="npz",  # 🔥 关键
    #     use_pca=False,
    #     batch_size=1
    # )
    # smpl_model = smplx.create(
    #     model_path="./data/models",  # directory containing `smpl/SMPL_NEUTRAL.npz`
    #     model_type="smpl",
    #     gender="neutral",
    #     ext="npz",  # key: tells smplx to load npz instead of pkl
    #     use_pca=False,
    #     batch_size=1
    # )


    for i in range(B):
        # imgname
        name = imgnames[i]
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        else:
            name = str(name)

        # vertices
        verts = labels[i]
        try:
            verts_list = verts.tolist()
        except Exception:
            verts_list = verts

        # pose
        pose_i = poses[i]
        try:
            pose_list = pose_i.tolist()
        except Exception:
            pose_list = pose_i

        # shape
        shape_i = shapes[i]
        try:
            shape_list = shape_i.tolist()
        except Exception:
            shape_list = shape_i

        # cam
        cam_i = cams[i]
        try:
            cam_list = cam_i.tolist()
        except Exception:
            cam_list = cam_i

        # verts_uv
        verts_uv_i = smpl_to_uv_single(pose_i, shape_i, cam_i, 32, 32)
        try:
            verts_uv_list = verts_uv_i.tolist()
        except Exception:
            verts_uv_list = verts_uv_i

        samples.append({
            "imgname": name,
            "vertices": verts_list,
            "pose": pose_list,
            "shape": shape_list,
            "cam": cam_list,
            "verts_uv": verts_uv_list   # ✅ 已生成 uv
        })

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets", help="Path to dataset directory")
    parser.add_argument("--imgname", default=None, help="Path to imgname.npy (overrides dataset-dir)")
    parser.add_argument("--labels", default=None, help="Path to contact_label.npy/contact_label_smplx.npy (overrides dataset-dir)")
    parser.add_argument("--out", default=None, help="Output path for samples.pth")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    imgname_path = args.imgname or os.path.join(dataset_dir, "imgname.npy")
    labels_path = args.labels or os.path.join(dataset_dir, "contact_label.npy") # contact_label_smplx.npy
    poses_path = os.path.join(dataset_dir, "pose.npy")
    shapes_path = os.path.join(dataset_dir, "shape.npy")
    cams_path = os.path.join(dataset_dir, "cam_k.npy")
    out_path = args.out or os.path.join(dataset_dir, "samples_smpl_cam2.pth")      # TODO: samples.pth

    imgnames = load_npy(imgname_path)       # (4380,)
    labels = load_npy(labels_path)          # (4380,6890)
    poses = load_npy(poses_path)
    shapes = load_npy(shapes_path)
    cams = load_npy(cams_path)

    samples = build_samples(imgnames, labels, poses, shapes, cams)

    # Ensure output dir exists
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    torch.save(samples, out_path)
    print(f"Saved {len(samples)} samples to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
