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

contact = np.load("contact_1_10.npy").squeeze()    # contact = np.random.rand(6890)      # ndarray: (6890,)

colors = np.ones((6890,4))*200      # Creates an RGBA color array for each vertex. Default color: [200, 200, 200, 200] → light gray.
colors[contact>0.1] = [0,0,255,255] # blue

mesh = trimesh.Trimesh(verts, faces, vertex_colors=colors)
mesh.show()

# save image
# scene = mesh.scene()
# png = scene.save_image(resolution=(1024,1024))
#
# with open("contact_4.png","wb") as f:
#     f.write(png)

# mesh.export("contact_4.ply")
# mesh.export("contact_4.obj")