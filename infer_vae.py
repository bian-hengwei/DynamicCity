"""
Inference script for saving VAE results.
Example: python infer_vae.py -n DynamicCityVAE --save_rollout --best
"""

import argparse
import warnings

from omegaconf import OmegaConf

from dynamic_city.trainer.vae_trainer import VAETrainer
from dynamic_city.utils.ckpt_utils import get_vae_ckpt, load_conf
from dynamic_city.utils.torch_utils import cleanup_dist, set_seed, set_tf32, setup_dist


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, required=True)
    parser.add_argument('--save_rollout', action='store_true')
    parser.add_argument('--save_voxel', default=None, type=int)  # number of results to be saved
    parser.add_argument('--save_layout', default=None, type=int)  # layout downsample rate
    parser.add_argument('--best', action='store_true')
    args = parser.parse_args()
    return args


def main():
    warnings.filterwarnings('ignore')

    # parse arguments and load conf
    args = get_args()
    conf = load_conf(args.name)

    # overwrite with inference conf
    OmegaConf.set_struct(conf, False)
    conf.trainer.auto_resume = True
    conf.dataset.batch_size = 1
    conf.dataset.valid_batch_size = 1
    conf.trainer.data_length = args.save_voxel if args.save_voxel is not None else -1
    conf.dataset.shuffle_dataset = args.save_voxel is not None
    if args.best:
        conf.trainer.resume_ckpt = get_vae_ckpt(conf.name)[0]

    # torch, ddp, and seed setup
    set_tf32(conf.trainer.tf32)
    rank, device = setup_dist()
    set_seed(conf.trainer.seed, conf.trainer.deterministic, rank)

    # inference
    trainer = VAETrainer(conf, device)
    trainer.predict(args.save_rollout, args.save_voxel, args.save_layout)

    # clean up
    cleanup_dist()


if __name__ == '__main__':
    main()
