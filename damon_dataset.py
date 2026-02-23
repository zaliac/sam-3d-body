# datasets/damon_dataset.py
import torch
# from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

class DamonDataset(Dataset):
    def __init__(self, samples, img_size=512):
        self.samples = samples
        self.img_size = img_size

        # self.transform = transforms.Compose([
        #     transforms.Resize((img_size, img_size)),
        #     transforms.ToTensor()
        # ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # img = cv2.imread(s["imgname"])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (self.img_size, self.img_size))
        # img = torch.from_numpy(img).float() / 255.
        # img = img.permute(2, 0, 1)
        # imgurl=os.path.abspath(s["imgname"])
        # img = Image.open(imgurl)      # s["imgname"]    .convert("RGB")
        # img = Image.open(s["imgname"]).convert('RGB')
        # img = transforms.ToTensor()(img)    # Tensor: (3,500,332):[[[0.67059,0.76078,0.82353,0.78431,0.71765,0.62353,0.45490,0.38431,0.41961,...],...],...]
        # img = self.transform(img)

        return {
            "id": idx,
            "image_path": s["imgname"],
            # "vertices": torch.tensor(s["smpl_vertices"]).float(),
            # TODO: add pose, shape
            "pose":s["pose"],           # (4384, 72)
            "shape":s["shape"],         # (4384, 10)
            "contact": torch.tensor(s['vertices']).long()     # "contact_labels"
        }
