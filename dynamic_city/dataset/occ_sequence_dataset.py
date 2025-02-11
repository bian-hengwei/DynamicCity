import random
from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import Dataset, default_collate

from dynamic_city.utils.data_utils import apply_augmentation, parse_semantic_dict


class OccSequenceDataset(ABC, Dataset):
    def __init__(self, dataset_conf, split, max_length=-1):
        self.dataset_conf = dataset_conf
        self.split = split

        self.augment = dataset_conf.augment
        self.sequence_length = dataset_conf.sequence_length
        self.spatial_size = dataset_conf.spatial_size
        self.num_classes = dataset_conf.num_classes

        if hasattr(dataset_conf, 'semantic_map'):
            self.semantic_map = parse_semantic_dict(dataset_conf.semantic_map).astype(np.int32)
        else:
            self.semantic_map = None

        self.sequences = self.get_sequences()  # OVERRIDE!

        if dataset_conf.shuffle_dataset:
            random.shuffle(self.sequences)

        if max_length > 0:
            self.sequences = self.sequences[:max_length]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        aug = self.split == 'train' and random.random() < self.augment
        aug_type = random.randint(0, 7) if aug else 0

        voxels, masks, paths = list(), list(), list()

        for path in self.sequences[index]:
            voxel, mask = self.get_voxel(path)  # OVERRIDE!
            voxels.append(voxel)
            masks.append(mask)
            paths.append(path)

        voxels, masks, paths = [default_collate(arr) for arr in [voxels, masks, paths]]
        voxels, masks, paths = apply_augmentation(voxels, masks, paths, aug_type)

        return {
            'voxels': voxels,
            'masks': masks,
            'paths': paths,
        }

    @abstractmethod
    def get_sequences(self):
        pass

    @abstractmethod
    def get_voxel(self, path):
        pass
