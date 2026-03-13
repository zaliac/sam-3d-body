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


def smpl_to_uv_single(pose, shape, cam, smpl_model, H_img=32, W_img=32):

    pose_t = torch.zeros(1,72).float()
    shape_t = torch.zeros(1,10).float()

    output = smpl_model(
        betas=shape_t,
        body_pose=pose_t[:,3:],
        global_orient=pose_t[:,:3],
        return_verts=True
    )

    verts = output.vertices[0]   # (6890,3)

    X = verts[:,0]
    Y = verts[:,1]

    # cam = [s, tx, ty]
    cam = torch.as_tensor(cam, dtype=torch.float32)

    if cam.ndim == 2:   # 如果传入的是3x3矩阵
        s = 2.0 / max(H_img, W_img)
        tx = 0.0
        ty = 0.0
    else:
        s  = cam[0]
        tx = cam[1]
        ty = cam[2]

    u = s * X + tx
    v = s * Y + ty

    verts_uv = torch.stack([u, v], dim=-1)

    # 保证 grid_sample 可用
    verts_uv = torch.clamp(verts_uv, -1, 1)

    return verts_uv.detach().cpu().numpy()




def weak_perspective_projection(verts, cam):
    """
    verts: (B,6890,3)
    cam: (B,3) -> [scale, tx, ty]

    return:
        uv: (B,6890,2)  in [-1,1]
    """

    X = verts[..., 0]
    Y = verts[..., 1]

    s = cam[:, 0].unsqueeze(1)
    tx = cam[:, 1].unsqueeze(1)
    ty = cam[:, 2].unsqueeze(1)

    u = s * X + tx
    v = s * Y + ty

    uv = torch.stack([u, v], dim=-1)

    return uv






def build_smpl_adjacency(smpl_model):

    faces = smpl_model.faces
    num_verts = smpl_model.v_template.shape[0]

    adj = np.zeros((num_verts, num_verts))

    for f in faces:
        i, j, k = f
        adj[i, j] = adj[j, i] = 1
        adj[j, k] = adj[k, j] = 1
        adj[k, i] = adj[i, k] = 1

    adj = adj + np.eye(num_verts)

    deg = np.sum(adj, axis=1)
    deg_inv = np.diag(1.0 / deg)

    adj = deg_inv @ adj

    return torch.tensor(adj, dtype=torch.float32)

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
    smpl_model = smplx.create(
        model_path="./data/models",  # directory containing `smpl/SMPL_NEUTRAL.npz`
        model_type="smpl",
        gender="neutral",
        ext="npz",  # key: tells smplx to load npz instead of pkl
        use_pca=False,
        batch_size=1
    )


    # -----------------------------
    # Build adjacency once
    # -----------------------------
    # adjacency = build_smpl_adjacency(smpl_model)        # Tensor(6890,6890)
    #
    # torch.save(adjacency, f"adjacency.pth")

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
        verts_uv_i = smpl_to_uv_single(pose_i, shape_i, cam_i, smpl_model,32, 32)   # pose_i, shape_i  change to standard smpl.
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
            "verts_uv": verts_uv_list   # [-1,1]
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
    out_path = args.out or os.path.join(dataset_dir, "samples_smpl_cam_standard2.pth")      # TODO: samples.pth

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
