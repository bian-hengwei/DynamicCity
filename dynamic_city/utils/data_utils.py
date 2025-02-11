import random
from enum import Enum

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from dynamic_city.utils.dist_utils import distributed


# dataloader
def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(dataset, batch_size, num_workers, seed=0, shuffle=True, drop_last=False):
    generator = torch.Generator()
    generator.manual_seed(seed)

    if distributed():
        sampler = DistributedSampler(dataset, seed=seed)
        shuffle = None
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        generator=generator,
        drop_last=drop_last,
        pin_memory=True,
    )


# occ sequence dataset
def apply_augmentation(voxels, masks, paths, aug_type=0):
    if aug_type & 1:  # Flip X
        voxels = torch.flip(voxels, dims=[1])
        masks = torch.flip(masks, dims=[1])
    if aug_type & 2:  # Flip Y
        voxels = torch.flip(voxels, dims=[2])
        masks = torch.flip(masks, dims=[2])
    if aug_type & 4:  # Flip T
        voxels = torch.flip(voxels, dims=[0])
        masks = torch.flip(masks, dims=[0])
        paths = paths[::-1]
    return voxels, masks, paths


# hexplane dataset
class Command(Enum):
    STATIC = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3


def get_command(traj_rot, angle_thr, forward_thr):
    rot_diff = traj_rot[:, 2] - traj_rot[0, 2]
    rot_diff = (rot_diff + np.pi) % (2 * np.pi) - np.pi
    if np.any(rot_diff > angle_thr):
        return Command.LEFT
    elif np.any(rot_diff < -angle_thr):
        return Command.RIGHT

    displacement = np.sqrt((traj_rot[:, 0] - traj_rot[0, 0]) ** 2 + (traj_rot[:, 1] - traj_rot[0, 1]) ** 2)
    if np.any(displacement > forward_thr):
        return Command.FORWARD
    else:
        return Command.STATIC


def get_trajectory(poses):
    return np.column_stack((poses[:, [3, 7]] - poses[0, [3, 7]], np.arctan2(poses[:, 4], poses[:, 0])))


# others
def parse_semantic_dict(semantic_dict):
    max_key = max(semantic_dict.keys())
    array = np.full(max_key + 1, 0, dtype=np.int64)
    for key, value in semantic_dict.items():
        array[key] = value
    return array


def convert_voxels_to_layouts(voxels, down_size):
    # Assumes that 10 means vehicle
    voxel_binary = (voxels == 10).to(torch.float32)  # T, X, Y, Z
    bev = voxel_binary.max(dim=3).values  # T, X, Y
    bev = bev.reshape(voxels.shape[0], 1, bev.shape[1], bev.shape[2])  # T, 1, X, Y
    layout = torch.nn.functional.max_pool2d(bev, down_size)  # T, 1, X, Y
    return layout[:, 0]  # T, X, Y


def parse_rollout_path(dataset_cfg, voxel_path):
    if dataset_cfg.dataset == 'carlasc':
        return (voxel_path
                .replace(dataset_cfg.data_path.rstrip('/') + '/', '')
                .replace(dataset_cfg.scene_folder.rstrip('/') + '/', '')
                .replace('.label', '.npz'))
    if dataset_cfg.dataset == 'occ3dw':
        return (voxel_path
                .replace(dataset_cfg.data_path.rstrip('/') + '/', '')
                .replace('.npz', '.npz'))
    raise NotImplementedError
