# train_damon.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from datasets.damon_dataset import DamonDataset
# from models.sam3d_damon import Sam3DDamon
# from losses.damon_loss import contact_loss, mesh_loss
import traceback
import torch.nn.functional as F
from damon_dataset import DamonDataset
# from sam3d_damon_old import Sam3DDamon
from sam3d_damon import Sam3DWithContact
from damon_loss import contact_loss, mesh_loss
import numpy as np
from sam_3d_body.utils import recursive_to
from torch.utils.tensorboard import SummaryWriter
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = Sam3DDamon().to(device)
model = Sam3DWithContact('checkpoints/sam-3d-body-dinov3/model.ckpt').to(device)       # checkpoints/sam-3d-body-dinov3

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-5     # lr=2e-5 lr=1e-4
)
accum_steps = 8
optimizer.zero_grad(set_to_none=True)

# ckpt = torch.load("sam3d_damon_1.pth", map_location=device)
# model.load_state_dict(ckpt, strict=False)    # model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False), model.load_state_dict(ckpt["model"], strict=False)
# optimizer.load_state_dict(ckpt["optimizer"])
# start_epoch = ckpt["epoch"] + 1

TRAIN_SAMPLES = torch.load('labels.pth', weights_only=False)     # 'samples_smpl_cam_special.pth'

dataset = DamonDataset(TRAIN_SAMPLES)
loader = DataLoader(dataset, batch_size=1)
output_folder = "./datasets/damon"

# criterion_contact = nn.BCELoss()
# Replace the weight and criterion lines:
pos_weight = torch.tensor(11.411, device=device)
criterion_contact = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


criterion_pose = torch.nn.MSELoss()
criterion_shape = torch.nn.MSELoss()

writer = SummaryWriter("logs/train")
global_step = 0

for epoch in range(20):
    model.train()
    for label in loader:
        # TODO: load batch by b["id"], then put to cuda
        i = label["id"].item()   # 0
        # if(i==81):
        #     pass
        try:
            path = f"{output_folder}/batch_{i}.pt"
            if os.path.exists(path):
                batch = torch.load(path, map_location="cpu", weights_only=False)
            else:
                print(f"Missing file: {path}")
                continue
            # batch = torch.load(f"{output_folder}/batch_{i}.pt", map_location="cpu", weights_only=False)
            num_img = batch['img'].shape[1]
            if num_img>1:
                batch['img'] = batch['img'][:, 0:1, :, :, :]
                batch['bbox'] = batch['bbox'][:, 0:1, :]
                batch['bbox_format'] = batch['bbox_format'][:1]
                batch['mask'] = batch['mask'][:, 0:1, :, :, :]
                batch['mask_score'] = batch['mask_score'][:, 0:1]
                batch['bbox_center'] = batch['bbox_center'][:, 0:1, :]
                batch['bbox_scale'] = batch['bbox_scale'][:, 0:1, :]
                batch['orig_bbox_scale'] = batch['orig_bbox_scale'][0:1, :]
                batch['bbox_expand_factor'] = batch['bbox_expand_factor'][:1]
                batch['ori_img_size'] = batch['ori_img_size'][:, 0:1, :]
                batch['img_size'] = batch['img_size'][:, 0:1, :]
                batch['input_size'] = batch['input_size'][0:1, :]
                batch['affine_trans'] = batch['affine_trans'][:, 0:1, :, :]
                batch['person_valid'] = batch['person_valid'][:, 0:1]

            if batch:
                label = recursive_to(label, device.type)
                batch = recursive_to(batch, device.type)

                # gt_c = label["contact"].float()
                gt_pose = label["pose"]
                gt_shape = label["shape"]

                out = model(batch, label)

                # loss = (
                #     contact_loss(out["contact"], gt_c)
                #     + 0.05 * mesh_loss(out["verts"], gt_v)
                # )
                # contact_probs = out["contact_probs"]        # (1,6890)

                # np.save(f"contact_1_{i}.npy", contact_probs.detach().cpu().numpy()) # for evaluate
                # loss_contact = criterion_contact(contact_probs, gt_c)   # (0.6930)     # contact_probs shape: (1,6890),gt_c shape:(1,6890)
                # loss_contact = criterion_contact(contact_probs, gt_c)  # contact_probs is now logits

                contact_logits = out["contact_logits"]  # [B,6890]
                gt_c = label["contact"].float().to(contact_logits.device)
                # loss_contact = criterion_contact(contact_logits, gt_c)
                # Use model-provided valid mask if available; otherwise derive from verts_uv
                valid_mask = out.get("valid_mask", None)
                if valid_mask is None:
                    verts_uv = out["verts_uv"]  # [B,6890,2]
                    valid_mask = (
                            (verts_uv[..., 0] >= -1.0) & (verts_uv[..., 0] <= 1.0) &
                            (verts_uv[..., 1] >= -1.0) & (verts_uv[..., 1] <= 1.0)
                    )
                valid_mask_f = valid_mask.float()

                # Elementwise BCE + mask
                bce_elem = F.binary_cross_entropy_with_logits(
                    contact_logits,
                    gt_c,
                    pos_weight=pos_weight,
                    reduction="none",
                )
                loss_contact = (bce_elem * valid_mask_f).sum() / valid_mask_f.sum().clamp(min=1.0)
                with torch.no_grad():
                    valid_mask_bool = valid_mask.bool()
                    pred_contact = contact_logits >= 0.0  # sigmoid(logit) >= 0.5
                    gt_contact_bin = gt_c >= 0.5

                    correct = (pred_contact == gt_contact_bin) & valid_mask_bool
                    contact_acc = correct.float().sum() / valid_mask_bool.float().sum().clamp(min=1.0)

                    # optional: useful for monitoring data quality
                    valid_ratio = valid_mask_bool.float().mean()

                # if(i%100==0):
                    # print("contact probs mean", torch.sigmoid(contact_probs).mean().item())
                    # print("contact probs mean", torch.sigmoid(contact_logits).mean().item())
                # ---- debug : 1 probes (every 50 steps) ----
                # if global_step % 50 == 0:
                #     with torch.no_grad():
                #         verts_uv = out["verts_uv"]  # [B,6890,2], expected in [-1,1]
                #         uv_oob = ((verts_uv < -1.0) | (verts_uv > 1.0)).any(dim=-1).float().mean().item()
                #
                #         probs = torch.sigmoid(contact_logits)
                #         print(
                #             f"[dbg step={global_step}] "
                #             f"loss_contact={loss_contact.item():.4f} "
                #             f"logit_mean={contact_logits.mean().item():.4f} "
                #             f"logit_std={contact_logits.std().item():.4f} "
                #             f"prob_mean={probs.mean().item():.4f} "
                #             f"gt_pos_rate={gt_c.mean().item():.4f} "
                #             f"uv_oob_rate={uv_oob:.4f}"
                #         )
                if global_step % 50 == 0:
                    with torch.no_grad():
                        probs = torch.sigmoid(contact_logits)
                        valid_ratio = valid_mask_f.mean().item()
                        uv_oob = 1.0 - valid_ratio
                        print(
                            f"[dbg step={global_step}] "
                            f"loss_contact={loss_contact.item():.4f} "
                            f"logit_mean={contact_logits.mean().item():.4f} "
                            f"logit_std={contact_logits.std().item():.4f} "
                            f"prob_mean={probs.mean().item():.4f} "
                            f"gt_pos_rate={gt_c.mean().item():.4f} "
                            f"valid_ratio={valid_ratio:.4f} "
                            f"uv_oob_rate={uv_oob:.4f}"
                        )

                # mhr = out["mhr"]
                # pred_pose = mhr["smpl_pose"]        # (1,72)
                #
                # device = pred_pose.device
                # gt_pose_tensor = torch.tensor([p.item() for p in gt_pose], device=device).unsqueeze(0)
                # gt_shape_tensor = torch.tensor([s.item() for s in gt_shape], device=device).unsqueeze(0)
                #
                # loss_pose = criterion_pose(pred_pose, gt_pose_tensor)   # (0.4064)
                #
                # pred_shape = mhr["smpl_shape"]
                # loss_shape = criterion_shape(pred_shape, gt_shape_tensor)   # (0.4595)
                mhr = out["mhr"]
                pred_pose = mhr["smpl_pose"]  # [B,72]
                pred_shape = mhr["smpl_shape"]  # [B,10]

                gt_pose_tensor = torch.as_tensor(label["pose"], dtype=torch.float32, device=pred_pose.device).view_as(
                    pred_pose)
                gt_shape_tensor = torch.as_tensor(label["shape"], dtype=torch.float32,
                                                  device=pred_shape.device).view_as(pred_shape)

                loss_pose = criterion_pose(pred_pose, gt_pose_tensor)
                loss_shape = criterion_shape(pred_shape, gt_shape_tensor)

                # loss = 1.0*loss_contact + 0.2*loss_pose + 0.2*loss_shape    # TODO: add loss weights

                # optimizer.zero_grad()
                # loss.backward()

                loss = 1.0 * loss_contact + 0.2 * loss_pose + 0.2 * loss_shape
                (loss / accum_steps).backward()

                # Step every accum_steps
                if (global_step + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                # ---- debug : 2 probes (every 50 steps) ----
                # if global_step % 50 == 0:
                #     total_norm_sq = 0.0
                #     for n, p in model.contact_head.named_parameters():
                #         if p.grad is not None:
                #             total_norm_sq += p.grad.data.norm(2).item() ** 2
                #     grad_norm = total_norm_sq ** 0.5
                #     print(f"[dbg step={global_step}] contact_head_grad_norm={grad_norm:.6f}")

                # ================= TensorBoard =================
                writer.add_scalars(
                    "Loss",
                    {
                        "total": loss.item(),
                        "contact": loss_contact.item(),
                        "pose": loss_pose.item(),
                        "shape": loss_shape.item(),
                    },
                    global_step
                )

                writer.add_scalars(
                    "Metrics",
                    {
                        "contact_accuracy": contact_acc.item(),
                        "valid_ratio": float(valid_ratio),
                    },
                    global_step
                )

                global_step += 1
        except Exception as e:
            print(f"error: [Epoch: {epoch} i: {i}]")
            print(e)
            traceback.print_exc()
    # torch.save(model.state_dict(), f"sam3d_damon_{epoch}.pth")
torch.save(model.state_dict(), f"sam3d_damon_20.pth")
