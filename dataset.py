import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BottleDataset(Dataset):

    def __init__(self, image_dir, pose_dir, ids):
        self.images = np.stack([cv2.imread(os.path.join(image_dir,
            f"{x}.png")) for x in ids]).transpose(0, 3, 1, 2)
        self.poses = np.stack([np.loadtxt(os.path.join(pose_dir,
            f"{x}.txt")) for x in ids])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx])
        pose = torch.from_numpy(self.poses[idx])

        return image, pose
