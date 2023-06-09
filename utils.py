import imageio
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def compute_psnr_from_mse(mse):
    return -10 * np.log(mse + 1e-8) / np.log(10)


def _get_pose(theta, phi, radius):
    pose = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, radius],
        [0, 0, 0, 1],
    ], dtype=torch.float)
    rot_theta = torch.eye(4)
    rot_theta[:3, :3] = torch.from_numpy(R.from_euler("y", -theta).as_matrix())
    rot_phi = torch.eye(4)
    rot_phi[:3, :3] = torch.from_numpy(R.from_euler("x", phi).as_matrix())
    rot = torch.tensor([
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float)
    pose = rot @ (rot_theta @ (rot_phi @ pose))

    return pose


def get_vis_poses(thetas, phi, radius):
    vis_poses = torch.stack([_get_pose(theta, phi, radius)
        for theta in thetas])

    return vis_poses


def save_video(images, output_path):
    imageio.mimwrite(output_path, images, fps=10, quality=8)


if __name__ == "__main__":
    # my_res = _get_pose(np.pi / 7, np.pi / 7, 4)
    # nerf_res = pose_spherical(180 / 7, 180 / 7, 4)
    # print(np.allclose(my_res, nerf_res))
    pass
