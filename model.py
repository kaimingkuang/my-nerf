import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MLPLayer(nn.Sequential):

    def __init__(self, in_feats, out_feats):
        layers = [
            nn.Linear(in_feats, out_feats),
            nn.BatchNorm1d(out_feats),
            nn.LeakyReLU(inplace=True)
        ]
        super().__init__(*layers)


class NeRF(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        sigma_mlp_chs = self.cfg.model.sigma_mlp_channels
        self.sigma_mlp = nn.Sequential(*[MLPLayer(in_ch, out_ch)
            for in_ch, out_ch in zip(sigma_mlp_chs[:-1], sigma_mlp_chs[1:])])
        self.sigma_head = nn.Sequential(*[
            nn.Linear(sigma_mlp_chs[-1], 1),
            nn.ReLU()
        ])

        rgb_mlp_chs = self.cfg.model.rgb_mlp_channels
        self.rgb_mlp = nn.Sequential(*[MLPLayer(in_ch, out_ch)
            for in_ch, out_ch in zip(rgb_mlp_chs[:-1], rgb_mlp_chs[1:])])
        self.rgb_head = nn.Sequential(*[
            nn.Linear(rgb_mlp_chs[-1], 3),
            nn.Sigmoid()
        ])

    def compute_rays(self, image_shape, int_mat, pose):
        h, w = image_shape
        xs, ys = torch.meshgrid(torch.linspace(0, h - 1, h),
            torch.linspace(0, w - 1, w), indexing="ij")
        xs, ys = xs.T, ys.T
        focal_x, focal_y = int_mat[0, 0], int_mat[1, 1]
        princ_x, princ_y = int_mat[0, 2], int_mat[1, 2]
        directions = torch.stack([
            (xs - princ_x) / focal_x,
            -(ys - princ_y) / focal_y,
            -torch.ones_like(xs)
        ], -1)
        rays_d = torch.einsum("hwm,mn->hwn", directions, pose[:3, :3].T)
        rays_o = pose[:3, -1].expand(rays_d.shape)

        return rays_o, rays_d

    def sample_random_coords(self, rays_o, rays_d, image, n_coords):
        # pos_coords = torch.argwhere(image.sum(dim=-1) > 0)
        # neg_coords = torch.argwhere(image.sum(dim=-1) == 0)
        # n_pos_coords = int(self.cfg.train.batch_size\
        #     * self.cfg.model.pos_pts_pct)
        # n_neg_coords = n_coords - n_pos_coords
        # rnd_pos_indices = np.random.choice(pos_coords.size()[0],
        #     size=(n_pos_coords, ), replace=False)
        # rnd_neg_indices = np.random.choice(neg_coords.size()[0],
        #     size=(n_neg_coords, ), replace=False)
        # rnd_pos_coords = pos_coords[rnd_pos_indices]
        # rnd_neg_coords = neg_coords[rnd_neg_indices]
        # rnd_coords = torch.cat([rnd_pos_coords, rnd_neg_coords], dim=0)
        h, w = image.size()[:2]
        coords = torch.stack(torch.meshgrid(torch.linspace(0, h - 1, h),
            torch.linspace(0, w - 1, w), indexing="ij"), dim=-1)
        coords = coords.reshape((-1, 2)).long()
        rnd_indices = np.random.choice(coords.size()[0],
            size=(n_coords, ), replace=False)
        rnd_coords = coords[rnd_indices]
        rays_o = rays_o[rnd_coords[:, 0], rnd_coords[:, 1]]
        rays_d = rays_d[rnd_coords[:, 0], rnd_coords[:, 1]]

        if image is not None:
            targets = image[rnd_coords[:, 0], rnd_coords[:, 1]]
            return rays_o, rays_d, targets
        else:
            return rays_o, rays_d

    def sample_points_on_rays(self, near, far, n_pts, rays_o, rays_d):
        b = rays_o.size(0)
        if self.cfg.model.perturb:
            bins = torch.linspace(near, far, n_pts + 1)
            lowers, uppers = bins[:-1], bins[1:]
            ts = torch.rand(b, n_pts) * (uppers - lowers) + lowers
        else:
            near = torch.ones((b, 1)) * near
            far = torch.ones((b, 1)) * far
            ts = torch.linspace(0, 1, n_pts)
            ts = near * (1 - ts) + far * ts
            ts = ts.expand((b, n_pts))
        pts_on_rays = rays_o[:, None, :] + rays_d[:, None, :]\
            * ts[..., None]

        return pts_on_rays, ts

    def compute_pos_encodings(self, pos, min_freq_pow, max_freq_pow):
        frequencies = torch.pow(2, torch.arange(min_freq_pow, max_freq_pow))
        sines = torch.cat([torch.sin(pos * freq)
            for freq in frequencies], -1)
        cosines = torch.cat([torch.cos(pos * freq)
            for freq in frequencies], -1)
        identities = pos
        encodings = torch.cat([sines, cosines, identities], dim=-1)

        return encodings

    def forward(self, pt_encs, view_encs):
        b, n = pt_encs.size()[:-1]
        pt_encs = pt_encs.reshape((b * n, -1))
        view_encs = view_encs[:, None, :].expand([-1, n, -1])\
            .reshape((b * n, -1))

        pt_feats = self.sigma_mlp(pt_encs)
        sigmas = self.sigma_head(pt_feats)

        feats = torch.cat([pt_feats, view_encs], dim=-1)
        rgbs = self.rgb_head(self.rgb_mlp(feats))

        rgbs = rgbs.reshape((b, n, -1))
        sigmas = sigmas.reshape((b, n))

        return rgbs, sigmas

    def render_volume(self, rgbs, sigmas, ts, dir_norms):
        deltas = torch.diff(ts, dim=-1)
        deltas = torch.cat([deltas, torch.ones(deltas.size(0), 1) * 1e10],
            dim=-1)
        deltas = (deltas * dir_norms).to(rgbs.device)
        alphas = 1 - torch.exp(-sigmas * deltas)
        weights = alphas * torch.cumprod(torch.cat([
            torch.ones((alphas.shape[0], 1), device=rgbs.device),
            1 - alphas + 1e-10], dim=-1), -1)[:, :-1]
        rgbs = torch.sum(weights[..., None] * rgbs, dim=1)

        return rgbs
