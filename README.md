# [WIP] Reimplementation of NeRF

This is a reimplementation of NeRF in PyTorch, transribed from [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch). Hopefully it is easier to understand from a beginner point-of-view.

## TODO
- Upload sample data of lego;
- Testing on 800x800 resolution;
- WandB configurations.

## Usage
Before running any training, you should log in your WandB to enable profiling.
```bash
python train.py --config=configs/lego.yaml
```