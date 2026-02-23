# train_damon.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from datasets.damon_dataset import DamonDataset
# from models.sam3d_damon import Sam3DDamon
# from losses.damon_loss import contact_loss, mesh_loss

from damon_dataset import DamonDataset
# from sam3d_damon_old import Sam3DDamon
from sam3d_damon import Sam3DWithContact
from damon_loss import contact_loss, mesh_loss
import numpy as np
from sam_3d_body.utils import recursive_to

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = Sam3DDamon().to(device)
model = Sam3DWithContact('checkpoints/sam-3d-body-dinov3/model.ckpt').to(device)       # checkpoints/sam-3d-body-dinov3

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-5
)

TRAIN_SAMPLES = torch.load('samples_smpl.pth')
dataset = DamonDataset(TRAIN_SAMPLES)
loader = DataLoader(dataset, batch_size=1)
output_folder = "./datasets/damon"
# criterion = nn.BCEWithLogitsLoss()
criterion_contact = nn.BCELoss()
criterion_pose = torch.nn.MSELoss()
criterion_shape = torch.nn.MSELoss()

for epoch in range(1):
    model.train()
    # total = 0

    for b in loader:
        # TODO: load batch by b["id"], then put to cuda
        i = b["id"].item()   # 0

        try:
            batch = torch.load(f"{output_folder}/batch_{i}.pt", map_location="cpu", weights_only=False)
            if batch:
                b = recursive_to(b, "cuda")
                batch = recursive_to(batch, "cuda")

                gt_c = b["contact"].float()
                gt_pose = b["pose"]
                gt_shape = b["shape"]

                # batch['img']=batch['img'].unsqueeze(1)

                out = model(batch)

                # loss = (
                #     contact_loss(out["contact"], gt_c)
                #     + 0.05 * mesh_loss(out["verts"], gt_v)
                # )
                # loss = contact_loss(out["contact_logits"], gt_c)       #    contact
                contact_probs = out["contact_probs"]
                loss_contact = criterion_contact(contact_probs, gt_c)   # (0.6930)     # contact_probs shape: (1,6890),gt_c shape:(1,6890)

                mhr = out["mhr"]
                pred_pose = mhr["smpl_pose"]        # (1,72)

                device = pred_pose.device
                gt_pose_tensor = torch.tensor([p.item() for p in gt_pose], device=device).unsqueeze(0)
                gt_shape_tensor = torch.tensor([s.item() for s in gt_shape], device=device).unsqueeze(0)

                loss_pose = criterion_pose(pred_pose, gt_pose_tensor)   # (0.4064)

                pred_shape = mhr["smpl_shape"]
                loss_shape = criterion_shape(pred_shape, gt_shape_tensor)   # (0.4595)

                loss = loss_contact + loss_pose + loss_shape    # TODO: add loss weights

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # total += loss.item()
                if(i%50 ==0):
                    print(f"[Epoch {epoch} i {i}] Loss: {loss:.4f}")
        except Exception:
            print(f"error: [Epoch: {epoch} i: {i}]")

    # print(f"[Epoch {epoch}] Loss: {total:.4f}")
    torch.save(model.state_dict(), f"sam3d_damon_{epoch}.pth")
# torch.save(model.state_dict(), f"sam3d_damon.pth")
