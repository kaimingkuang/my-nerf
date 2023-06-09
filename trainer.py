import os
from datetime import datetime

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm

from model import NeRF
from utils import compute_psnr_from_mse, get_vis_poses, save_video


class Trainer:

    def __init__(self, cfg):
        self.cfg = cfg
        self.load_data()
        self.image_shape = self.images_train.size()[1:-1]

        self.model = NeRF(cfg).cuda()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(),
            self.cfg.train.max_lr)
        gamma = (self.cfg.train.min_lr / self.cfg.train.max_lr)\
            ** (1 / self.cfg.train.n_iters)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,
            gamma)

        # camera poses for visualization
        thetas = np.linspace(-np.pi, np.pi, self.cfg.eval.n_vis_poses + 1)
        phi = eval(self.cfg.eval.vis_phi)
        self.vis_poses = get_vis_poses(thetas, phi, self.cfg.eval.vis_radius)

        self.cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = f"logs/{self.cur_time}"
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(f"{self.log_dir}/videos", exist_ok=True)
        os.makedirs(f"{self.log_dir}/weights", exist_ok=True)

    def load_data(self):
        data = np.load(self.cfg.data.path, allow_pickle=True).item()
        self.images_train = torch.from_numpy(data["train"]["images"])
        self.poses_train = torch.from_numpy(data["train"]["poses"])
        self.images_val = torch.from_numpy(data["val"]["images"])
        self.poses_val = torch.from_numpy(data["val"]["poses"])
        self.images_test = torch.from_numpy(data["test"]["images"])
        self.poses_test = torch.from_numpy(data["test"]["poses"])
        focal = data["train"]["focal"]
        h, w = self.images_train.shape[2:]
        self.intrinsics = torch.tensor([
            [focal, 0, 0.5 * w],
            [0, focal, 0.5 * h],
            [0, 0, 1]
        ])

    def sample_image(self):
        rnd_img_idx = np.random.choice(self.images_train.shape[0])
        image = self.images_train[rnd_img_idx]
        pose = self.poses_train[rnd_img_idx]

        return image, pose

    def run_nerf(self, pose, image=None, training=True):
        # compute rays_o and rays_d
        rays_o, rays_d = self.model.compute_rays(self.image_shape,
            self.intrinsics, pose)

        # sample random coordinates calculate view directions
        if training:
            rays_o, rays_d, targets = self.model.sample_random_coords(rays_o,
                rays_d, image, self.cfg.train.batch_size)
        else:
            h, w = rays_o.size()[:-1]
            rays_o = rays_o.reshape((h * w, -1))
            rays_d = rays_d.reshape((h * w, -1))
            if image is not None:
                targets = image.reshape((h * w, -1))
        # normalize rays_d to get view directions
        views = F.normalize(rays_d, dim=-1)
        if image is not None:
            targets = targets.cuda()

        # sample points on rays
        pts_on_rays, ts = self.model.sample_points_on_rays(
            self.cfg.model.near,
            self.cfg.model.far,
            self.cfg.model.n_pts_on_ray,
            rays_o,
            rays_d
        )

        # convert pts on rays and view directions to position encodings
        pt_encodings = self.model.compute_pos_encodings(pts_on_rays,
            self.cfg.model.min_freq_pow_pts,
            self.cfg.model.max_freq_pow_pts)
        view_encodings = self.model.compute_pos_encodings(views,
            self.cfg.model.min_freq_pow_view,
            self.cfg.model.max_freq_pow_view)

        # send encodings to network inference and get RGB pred
        if training:
            rgbs, sigmas = self.model(pt_encodings.cuda(),
                view_encodings.cuda())
        else:
            batch_size = self.cfg.train.batch_size
            total_size = pt_encodings.size(0)
            rgbs = []
            sigmas = []
            for i in range(int(np.ceil(total_size / batch_size))):
                beg, end = i * batch_size, (i + 1) * batch_size
                res = self.model(pt_encodings[beg:end].cuda(),
                    view_encodings[beg:end].cuda())
                rgbs.append(res[0])
                sigmas.append(res[1])
            rgbs = torch.cat(rgbs)
            sigmas = torch.cat(sigmas)

        # volume rendering using RGB and sigma outputs from the model
        dir_norms = torch.norm(rays_d, dim=-1, keepdim=True)
        rgbs = self.model.render_volume(rgbs, sigmas, ts, dir_norms)

        if image is not None:
            return rgbs, targets
        else:
            return rgbs

    @torch.no_grad()
    def visualize(self, step):
        self.model.eval()
        # output visualization
        rgbs = []
        for pose in tqdm(self.vis_poses):
            rgbs.append(self.run_nerf(pose, training=False).cpu().numpy())
        b, (h, w) = len(rgbs), self.image_shape
        rgbs = np.stack(rgbs).reshape((b, h, w, -1))

        save_video(rgbs, f"{self.log_dir}/videos/step_{step}.mp4")
        wandb.log({"video": wandb.Video(rgbs, fps=10)})

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        # evaluate on val_images
        avg_loss = 0
        avg_psnr = 0
        for image, pose in zip(self.images_val, self.poses_val):
            rgbs, targets = self.run_nerf(pose, image, training=False)
            loss = self.criterion(rgbs, targets)
            psnr = compute_psnr_from_mse(loss.cpu().item())
            avg_loss += loss.cpu().item()
            avg_psnr += psnr

        avg_loss /= self.images_val.size(0)
        avg_psnr /= self.images_val.size(0)

        return avg_loss, avg_psnr

    def train(self):
        if not self.cfg.debug:
            wandb_cfg = OmegaConf.load("wandb_cfg.yaml")
            wandb.login(key=wandb_cfg.key)
            wandb.init(
                config=self.cfg,
                entity=wandb_cfg.entity,
                project=wandb_cfg.project,
                name=self.cur_time
            )
        else:
            self.cfg.eval.freq = 1

        best_psnr = 0
        for i in range(self.cfg.train.n_iters):
            self.model.train()
            # sample one image
            image, pose = self.sample_image()

            rgbs, targets = self.run_nerf(pose, image, training=True)

            # calculate loss and PSNR
            loss = self.criterion(rgbs, targets)
            psnr = compute_psnr_from_mse(loss.cpu().item())

            # backprop and network update
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            metrics = {
                "loss/train": loss.cpu().item(),
                "psnr/train": psnr,
                "step": i
            }

            # evaluation
            if (i + 1) % self.cfg.eval.freq == 0:
                self.visualize(step=i)
                val_loss, val_psnr = self.evaluate()
                metrics.update({
                    "loss/val": val_loss,
                    "psnr/val": val_psnr
                })

                print(f"Iter {i}\t\tLoss={loss:.4f}|{val_loss:.4f}, PSNR={psnr:.4f}|{val_psnr:.4f}")

                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    torch.save(self.model.state_dict(),
                        os.path.join(self.log_dir, "weights/model_best.pth"))
                torch.save(self.model.state_dict(),
                    os.path.join(self.log_dir, f"weights/model_{i}.pth"))

            if not self.cfg.debug:
                wandb.log(metrics)

        wandb.close()


if __name__ == "__main__":
    wandb_cfg = OmegaConf.load("wandb_cfg.yaml")
    wandb.login(key=wandb_cfg.key)
    wandb.init(
        # config=self.cfg,
        entity=wandb_cfg.entity,
        project=wandb_cfg.project,
        name="test"
    )
    frames = np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8)
    wandb.log({"video": wandb.Video(frames, fps=4)})
