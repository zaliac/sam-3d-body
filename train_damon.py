# train_damon.py
import torch
from torch.utils.data import DataLoader
# from datasets.damon_dataset import DamonDataset
# from models.sam3d_damon import Sam3DDamon
# from losses.damon_loss import contact_loss, mesh_loss

from damon_dataset import DamonDataset
# from sam3d_damon_old import Sam3DDamon
from sam3d_damon import Sam3DWithContact
from damon_loss import contact_loss, mesh_loss
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = Sam3DDamon().to(device)
model = Sam3DWithContact('checkpoints/sam-3d-body-dinov3/model.ckpt').to(device)       # checkpoints/sam-3d-body-dinov3

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-5
)

TRAIN_SAMPLES = torch.load('samples_20.pth')
dataset = DamonDataset(TRAIN_SAMPLES)
loader = DataLoader(dataset, batch_size=2, shuffle=True)
output_folder = "./datasets/damon"
for epoch in range(25):
    model.train()
    total = 0

    for b in loader:


        # TODO: load batch by b["id"], then put to cuda
        i = b["id"]
        batch = np.load(f"{output_folder}/batch_{i}.pt", allow_pickle=True)
        if batch:
            # batch = {k: v.to(device) for k, v in b.items()}  # move each tensor
            b = {k: v.to(device) for k, v in b.items()}
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.cuda()
            # img = b["image"].to(device)     # Tensor: (2,3,512,512)
            # gt_v = b["vertices"].to(device)
            # gt_c = b["contact"].to(device)
            # gt_c = batch["contact"]
            gt_c = b["contact"]


            # batch['img']=batch['img'].unsqueeze(1)

            out = model(batch)

            # loss = (
            #     contact_loss(out["contact"], gt_c)
            #     + 0.05 * mesh_loss(out["verts"], gt_v)
            # )
            loss = contact_loss(out["contact_logits"], gt_c)       #    contact

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()

    print(f"[Epoch {epoch}] Loss: {total:.4f}")
    torch.save(model.state_dict(), f"sam3d_damon_{epoch}.pth")
