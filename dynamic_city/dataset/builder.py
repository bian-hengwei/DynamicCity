from dynamic_city.dataset.carlasc import CarlaSCHexPlaneDataset, CarlaSCOccSequenceDataset
from dynamic_city.utils.data_utils import get_dataloader


def get_occ_sequence_dataloaders(dataset_conf, conf=None):
    max_length = conf.trainer.data_length if conf else -1
    seed = conf.trainer.seed if conf else 0

    dataset_class = {
        'carlasc': CarlaSCOccSequenceDataset,
    }[dataset_conf.dataset]

    train_dataset = dataset_class(dataset_conf, 'train', max_length)
    valid_dataset = dataset_class(dataset_conf, 'valid', max_length)

    train_dataloader = get_dataloader(
        train_dataset, dataset_conf.batch_size,
        dataset_conf.num_workers, seed, shuffle=True
    )
    valid_dataloader = get_dataloader(
        valid_dataset, dataset_conf.valid_batch_size,
        dataset_conf.num_workers, seed, shuffle=False
    )

    return train_dataloader, valid_dataloader


def get_hexplane_dataloaders(dit_conf, vae_conf, split='train'):
    dataset_class = {
        'carlasc': CarlaSCHexPlaneDataset,
    }[vae_conf.dataset.dataset]

    batch_size = {
        'train': dit_conf.dataset.batch_size,
        'valid': dit_conf.dataset.valid_batch_size,
    }[split]

    shuffle = {
        'train': True,
        'valid': False,
    }[split]

    return get_dataloader(
        dataset_class(dit_conf, vae_conf, split), batch_size,
        dit_conf.dataset.num_workers, dit_conf.trainer.seed, shuffle=shuffle, drop_last=True
    )
