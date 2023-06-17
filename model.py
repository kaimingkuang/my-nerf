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
        
    def hierarchical_sample(self, ts, weights, n_pts, pts_coarse,
            rays_o, rays_d):
        # sample points according to weights given in fine mode,
        # aka, hierarchical volume rendering (Section 5.2)
        bins = 0.5 * (ts[..., 1:] + ts[..., :-1])
        weights = weights[..., 1:-1] + 1e-5
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)

        # Take uniform samples
        u = torch.rand(list(cdf.shape[:-1]) + [n_pts])

        # Invert CDF
        u = u.contiguous()
        indices = torch.searchsorted(cdf, u, right=True)
        lowers = torch.max(torch.zeros_like(indices - 1), indices - 1)
        uppers = torch.min((cdf.shape[-1] - 1) * torch.ones_like(indices),
            indices)
        bounds = torch.stack([lowers, uppers], -1)

        matched_shape = [bounds.shape[0], bounds.shape[1], cdf.shape[-1]]
        cdf = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2,
            bounds)
        bins = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2,
            bounds)

        denom = (cdf[..., 1] - cdf[..., 0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        ts = (u - cdf[...,0]) / denom
        ts_fine = (bins[...,0] + ts * (bins[...,1] - bins[...,0])).detach()

        pts_fine = rays_o[:, None, :] + rays_d[:, None, :]\
            * ts_fine[..., None]

        return pts_fine, ts_fine

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

        return rgbs, weights.cpu()
