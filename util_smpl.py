import torch
import smplx

def smpl_to_uv_batch(pose, shape, K, H_img, W_img, smpl_model_path, gender="neutral", device="cpu"):
    """
    Batch SMPL to UV coordinates for grid_sample.

    Args:
        pose: (B,72) SMPL pose in degrees
        shape: (B,10) SMPL shape
        K: (B,3,3) camera intrinsics
        H_img, W_img: image height/width
        smpl_model_path: path to SMPL model
        gender: "neutral"
        device: "cpu" or "cuda"

    Returns:
        verts_uv: (B,6890,2) in [-1,1] for grid_sample
    """
    B = pose.shape[0]

    # 1 init the smpl model
    smpl_model = smplx.create(
        model_path=smpl_model_path,
        model_type="smpl",
        gender=gender,
        batch_size=B
    ).to(device)

    # 2 SMPL forward
    output = smpl_model(
        betas=shape.to(device),
        body_pose=pose[:,3:].to(device),
        global_orient=pose[:,:3].to(device),
        return_verts=True
    )

    # Get vertices in meters
    verts_3d = output.vertices * 0.001  # mm to m  * 0.001

    # Project to image pixels
    X = verts_3d[..., 0]
    Y = verts_3d[..., 1]
    Z = verts_3d[..., 2].clamp(min=1e-6)  # Avoid division by zero

    fx = K[:, 0, 0].unsqueeze(1)
    fy = K[:, 1, 1].unsqueeze(1)
    cx = K[:, 0, 2].unsqueeze(1)
    cy = K[:, 1, 2].unsqueeze(1)

    u = fx * (X / Z) + cx
    v = fy * (-Y / Z) + cy  # Flip for image v increasing up or down

    # Normalize to [-1, 1] for grid_sample (assuming feature map matches image scale)
    u_norm = 2 * (u / W_img) - 1
    v_norm = 2 * (v / H_img) - 1

    verts_uv = torch.stack([u_norm, v_norm], dim=-1)

    return verts_uv



def project_vertices(vertices, K):
    """
    vertices: [B,6890,3]
    K: [B,3,3]
    return: UV [B,6890,2]
    """

    B, N, _ = vertices.shape

    x = vertices[:,:,0]
    y = vertices[:,:,1]
    z = vertices[:,:,2].clamp(min=1e-6)

    fx = K[:,0,0].unsqueeze(1)
    fy = K[:,1,1].unsqueeze(1)
    cx = K[:,0,2].unsqueeze(1)
    cy = K[:,1,2].unsqueeze(1)

    u = fx * (x / z) + cx
    v = fy * (y / z) + cy

    uv = torch.stack([u,v], dim=-1)

    return uv


def smpl_to_uv(pose, shape, camera, smpl_model):

    global_orient = pose[:, :3]
    body_pose = pose[:, 3:]

    output = smpl_model(
        betas=shape,
        body_pose=body_pose,
        global_orient=global_orient,
        return_verts=True
    )

    vertices = output.vertices

    uv = project_vertices(vertices, camera)

    return vertices, uv
'''

# example
verts_uv = smpl_to_uv_batch(
    pose=pose,           # (B,72)
    shape=shape,         # (B,10)
    K=K,                 # (B,3,3)
    H_img=H_img,
    W_img=W_img,
    smpl_model_path="path_to_smpl_model",
    gender="neutral",
    device="cuda"
)

# 直接用于你的 ContactHeadSMPL
contact_prob = contact_head(feat_map, verts_uv, adjacency)


{
    "imgname": "image_0001.jpg",
    "vertices": [...],   # 6890
    "pose": [...],       # 72
    "shape": [...],      # 10
    "cam": [[fx,0,cx],[0,fy,cy],[0,0,1]]  # (3,3)
}

'''
