import torch
import smplx

def smpl_to_uv_batch(pose, shape, K, H_img, W_img, smpl_model_path, gender="neutral", device="cpu"):
    """
    Batch 版 SMPL -> verts_uv 生成函数

    Args:
        pose: (B,72) SMPL pose
        shape: (B,10) SMPL shape
        K: (B,3,3) 相机内参
        H_img, W_img: 原图尺寸
        smpl_model_path: SMPL 模型路径
        gender: "neutral", "male", "female"
        device: "cuda" or "cpu"

    Returns:
        verts_uv: (B,6890,2) in [-1,1]
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
    verts_3d = output.vertices  # (B,6890,3)

    # 3 project to pixel coordination
    X = verts_3d[...,0]
    Y = verts_3d[...,1]
    Z = verts_3d[...,2].clamp(min=1e-6)

    fx = K[:,0,0].unsqueeze(1)  # (B,1)
    fy = K[:,1,1].unsqueeze(1)
    cx = K[:,0,2].unsqueeze(1)
    cy = K[:,1,2].unsqueeze(1)

    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    uv_pixels = torch.stack([u,v], dim=-1)  # (B,6890,2)

    # 4 normalize to [-1,1]，for grid_sample
    u_norm = 2 * (uv_pixels[...,0] / (W_img - 1)) - 1
    v_norm = 2 * (uv_pixels[...,1] / (H_img - 1)) - 1
    verts_uv = torch.stack([u_norm, v_norm], dim=-1)

    return verts_uv


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
