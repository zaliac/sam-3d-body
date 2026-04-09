import os
import torch

labels = torch.load("labels.pth", weights_only=False)

filtered = []
for idx, sample in enumerate(labels):
    batch_path = f"./datasets/damon/batch_{idx}.pt"
    if os.path.exists(batch_path):
        filtered.append(sample)
    else:
        print(f"Removing missing sample idx={idx}, file={batch_path}")

torch.save(filtered, "labels_filtered.pth")
print(f"Saved {len(filtered)} samples to labels_filtered.pth")