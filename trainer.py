import os

import numpy as np
from torch.utils.data import DataLoader

from dataset import BottleDataset


class Trainer:

    def __init__(self, cfg):
        self.cfg = cfg
        self.image_dir = cfg.data.image_dir
        self.pose_dir = cfg.data.pose_dir
        self.intrinsics = np.loadtxt(cfg.data.intrinsics)

        self.train_ids = [x.replace(".png", "") for x
            in os.listdir(self.image_dir) if "train" in x]
        self.val_ids = [x.replace(".png", "") for x
            in os.listdir(self.image_dir) if "val" in x]
        
        self.ds_train = BottleDataset(self.image_dir, self.pose_dir,
            self.train_ids)
        self.ds_val = BottleDataset(self.image_dir, self.pose_dir,
            self.val_ids)


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("configs/bottles.yaml")
    trainer = Trainer(cfg)
    sample = trainer.ds_train[0]
    print(1)