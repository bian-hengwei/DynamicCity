from pathlib import Path

import numpy as np

from dynamic_city.dataset.hexplane_dataset import HexPlaneDataset
from dynamic_city.dataset.occ_sequence_dataset import OccSequenceDataset
from dynamic_city.utils.data_utils import get_trajectory


class CarlaSCOccSequenceDataset(OccSequenceDataset):
    def __init__(self, dataset_conf, split, max_length=-1):
        super().__init__(dataset_conf, split, max_length)

    def get_sequences(self):
        sequences = list()

        data_root = Path(self.dataset_conf.data_path) / self.dataset_conf.split[self.split]
        scene_roots = [p / self.dataset_conf.scene_folder for p in data_root.iterdir()]

        skip = self.dataset_conf.skip
        span = self.dataset_conf.sequence_length * (skip + 1) - skip
        overlap = span - 1

        for scene_root in scene_roots:
            scene_files = sorted(list(map(str, scene_root.glob(self.dataset_conf.data_pattern))))
            num_sequences = len(scene_files) - overlap
            sequences.extend([scene_files[s: s + span: skip + 1] for s in range(num_sequences)])

        return sequences

    def get_voxel(self, path):
        voxel = np.fromfile(path, dtype=np.uint32).reshape(self.spatial_size)
        if self.semantic_map is not None:
            voxel = self.semantic_map[voxel]
        valid = np.fromfile(path.replace('label', 'bin'), dtype=np.float32).reshape(self.spatial_size)
        invalid = (valid == 0).astype(np.uint8)
        return voxel, invalid


class CarlaSCHexPlaneDataset(HexPlaneDataset):
    def __init__(self, dit_conf, vae_conf, split):
        super().__init__(dit_conf, vae_conf, split)

    def prepare(self):
        hexplanes, conditions, trajectories, turns = list(), list(), list(), list()
        hexplane_roots = list(self.data_folder.iterdir())

        angle_thr = self.sequence_length * self.dit_conf.dataset.angle_thr_mul

        for hexplane_folder in hexplane_roots:
            hexplane_files = sorted(list(map(str, hexplane_folder.glob('*.npz'))))
            poses = np.loadtxt(
                Path(self.vae_conf.dataset.data_path) / Path(*hexplane_folder.parts[-2:]) /
                self.vae_conf.dataset.scene_folder.split('/')[0] / 'poses.txt'
            )
            for i in range(len(hexplane_files)):
                hexplane_path = Path(hexplane_files[i])
                hexplane_number_str = hexplane_path.stem
                hexplane_number = int(hexplane_number_str)

                if self.hex_cond:
                    condition_number = hexplane_number - self.sequence_length
                    condition_number_str = str(condition_number).zfill(len(hexplane_number_str))
                    condition_path = hexplane_path.parent / f'{condition_number_str}.npz'
                    if not condition_path.exists():
                        continue
                else:
                    condition_path = None

                if hexplane_number + self.sequence_length > len(poses):
                    continue
                trajectory = get_trajectory(poses[hexplane_number: hexplane_number + self.sequence_length])

                angle_diff = abs(trajectory[0][2] - trajectory[-1][2])
                turn = angle_diff > angle_thr

                hexplanes.append(hexplane_path)
                conditions.append(condition_path)
                trajectories.append(trajectory)
                turns.append(turn)

        return hexplanes, conditions, trajectories, turns
