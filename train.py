import random
from argparse import ArgumentParser

import numpy as np
import torch
from omegaconf import OmegaConf

from trainer import Trainer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    return args


def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.debug = args.debug

    set_rng_seed(42)

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
