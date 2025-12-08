import os, glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class SingleImageDataset(Dataset):
    def __init__(self, root_glob):
        self.paths = sorted(glob.glob(root_glob))
        assert len(self.paths) > 0, f"No images found for glob: {root_glob}"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # C,H,W
        name = os.path.splitext(os.path.basename(p))[0]
        return {"a": img, "name": name}
