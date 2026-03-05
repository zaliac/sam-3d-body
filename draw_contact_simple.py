import torch
import smplx
import trimesh
import numpy as np

model = smplx.create("./data/models", model_type="smpl")

pose = torch.zeros(1,72)
betas = torch.zeros(1,10)

out = model(betas=betas, body_pose=pose[:,3:], global_orient=pose[:,:3])

verts = out.vertices[0].detach().cpu().numpy()
faces = model.faces

contact = np.random.rand(6890)

colors = np.ones((6890,4))*200
colors[contact>0.5] = [0,0,255,255]

mesh = trimesh.Trimesh(verts, faces, vertex_colors=colors)
mesh.show()