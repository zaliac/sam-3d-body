import torch
from util_smpl import smpl_to_uv_batch  # Assumes smplx is installed

samples = torch.load('samples_smpl_cam_standard2.pth')

# Target resized size (your network input)
target_H, target_W = 512, 512

for i, s in enumerate(samples):
    K = torch.tensor(s['cam']).unsqueeze(0)  # (1,3,3)
    cx, cy = K[0, 0, 2].item(), K[0, 1, 2].item()

    # Approximate original size (cx ≈ W/2, cy ≈ H/2)
    orig_W = 2 * cx
    orig_H = 2 * cy

    # Scaling factors for resizing
    scale_x = target_W / orig_W
    scale_y = target_H / orig_H

    # Scale intrinsics to match resized image
    K_scaled = K.clone()
    K_scaled[0, 0, 0] *= scale_x  # fx
    K_scaled[0, 1, 1] *= scale_y  # fy
    K_scaled[0, 0, 2] *= scale_x  # cx
    K_scaled[0, 1, 2] *= scale_y  # cy

    pose = torch.tensor(s['pose']).unsqueeze(0)  # (1,72)
    shape = torch.tensor(s['shape']).unsqueeze(0)  # (1,10)

    # Project with scaled K and target size
    verts_uv = smpl_to_uv_batch(
        pose=pose,
        shape=shape,
        K=K_scaled,
        H_img=target_H,
        W_img=target_W,
        smpl_model_path="./data/models/smpl/SMPL_NEUTRAL.pkl",
        gender="neutral",
        device="cpu"
    )

    s['verts_uv'] = verts_uv.squeeze(0).detach().numpy()

    # Optional: Print progress/range for first few
    if i < 5:
        uv = torch.tensor(s['verts_uv'])
        print(f'Sample {i}: verts_uv min {uv.min().item():.3f} max {uv.max().item():.3f}')

torch.save(samples, 'samples_smpl_cam_standard2_uv_corrected.pth')
print("Saved samples with corrected verts_uv spanning [-1,1] properly")