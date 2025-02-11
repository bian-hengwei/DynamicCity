"""
Main script for training VAE / DiT models.
Example: python train.py VAE carlasc name=DynamicCityVAE
"""

import importlib
import sys
import warnings
from pathlib import Path

import hydra

from dynamic_city.utils.ckpt_utils import save_conf
from dynamic_city.utils.torch_utils import cleanup_dist, set_seed, set_tf32, setup_dist

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # parse arguments
    assert len(sys.argv) >= 3, 'Usage: python train.py trainer_name conf_path [overrides]'
    trainer_name = sys.argv[1]
    conf_path = sys.argv[2]
    overrides = sys.argv[3:]

    # parse carlasc as conf/vae/carlasc.yaml
    if not conf_path.endswith('.yaml'):
        conf_path += '.yaml'
    if not Path(conf_path).exists():
        conf_path = f'conf/{trainer_name.lower().replace("trainer", "")}/{conf_path}'
    conf_path = Path(conf_path)
    assert conf_path.exists(), f'Invalid config path: {str(conf_path)}'
    conf_root = conf_path.parent
    conf_name = conf_path.name

    # initialize hydra and load config
    hydra.initialize(config_path=str(conf_root), version_base=None)
    conf = hydra.compose(config_name=str(conf_name), overrides=overrides)

    # import trainer module
    module = importlib.import_module(f'dynamic_city.trainer')
    Trainer = getattr(module, trainer_name, None) or getattr(module, f'{trainer_name}Trainer', None)
    assert Trainer, f'Invalid trainer: {trainer_name}'

    # torch, ddp, and seed setup
    set_tf32(conf.trainer.tf32)
    rank, device = setup_dist()
    set_seed(conf.trainer.seed, conf.trainer.deterministic, rank)

    # save and print config
    save_conf(conf, print_conf=False)

    # train
    trainer = Trainer(conf, device)
    trainer.fit()

    # clean up
    cleanup_dist()
