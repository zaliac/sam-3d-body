import torch
import smplx
import numpy as np
B = 1

smpl_model = smplx.create(
    model_path="./data/models",
    model_type="smpl",
    gender="neutral",
    ext="npz",
    use_pca=False,
    batch_size=B
)

pose = torch.zeros(B, 72)
betas = torch.zeros(B, 10)

with torch.no_grad():
    output = smpl_model(
        betas=betas,
        global_orient=pose[:, :3],
        body_pose=pose[:, 3:],
        return_verts=True
    )

verts = output.vertices  # (B, 6890, 3)
verts_np = verts.cpu().numpy()  # verts.squeeze(0).cpu().numpy(): ->(6890,3)
np.save("./datasets/smpl_standard_vertices.npy", verts_np)
print(verts.shape)
print(verts)