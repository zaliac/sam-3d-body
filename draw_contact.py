import torch
import smplx
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    TexturesVertex
)
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# SMPL MODEL
# -----------------------------
model_path = "./data/models/smpl"
smpl = smplx.create(
    model_path,
    model_type='smpl',
    gender='neutral',
    batch_size=1
).to(device)

# pose (72) and shape (10)
pose = torch.zeros(1, 72).to(device)
betas = torch.zeros(1, 10).to(device)

output = smpl(
    betas=betas,
    body_pose=pose[:, 3:],
    global_orient=pose[:, :3],
    return_verts=True
)

vertices = output.vertices  # (1, 6890, 3)
faces = torch.tensor(smpl.faces.astype(np.int64)).unsqueeze(0).to(device)

# -----------------------------
# CONTACT LABELS
# -----------------------------
contact = torch.rand(6890).to(device)  # example contact labels 0~1

# threshold example
contact_binary = (contact > 0.5).float()

# -----------------------------
# VERTEX COLORS
# -----------------------------
# original mesh color (skin-like)
base_color = torch.tensor([0.7, 0.7, 0.7]).to(device)

colors = base_color.repeat(6890,1)

# blue color
blue = torch.tensor([0,0,1.0]).to(device)

colors = torch.where(
    contact_binary.unsqueeze(1) == 1,
    blue,
    colors
)

colors = colors.unsqueeze(0)

textures = TexturesVertex(verts_features=colors)

# -----------------------------
# BUILD MESH
# -----------------------------
mesh = Meshes(
    verts=vertices,
    faces=faces,
    textures=textures
)

# -----------------------------
# RENDER
# -----------------------------
cameras = FoVPerspectiveCameras(device=device)

lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1
)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device,
        cameras=cameras,
        lights=lights
    )
)

images = renderer(mesh)

plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off")
plt.show()