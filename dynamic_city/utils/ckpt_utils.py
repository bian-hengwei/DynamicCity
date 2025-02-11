from pathlib import Path

from omegaconf import OmegaConf

import dynamic_city.utils.constants as C
from dynamic_city.utils.dist_utils import func_rank_0


def _get_ckpts_sorted(pattern, func=None):
    matches = list(Path(C.CKPT_ROOT).glob(pattern))
    if len(matches) == 0:
        return [None]
    if func is not None:
        matches.sort(key=func, reverse=True)
    return matches


def get_vae_ckpt(prefix, epoch='*'):
    return _get_ckpts_sorted(f'{prefix}*/{epoch}_mIoU_*.ckpt', lambda x: float(x.stem.split('_')[2]))


def get_dit_ckpt(prefix, step=-1):
    return _get_ckpts_sorted(f'{prefix}*/*{"" if step == -1 else step}.ckpt', lambda x: int(x.stem))


def get_latest_ckpt(prefix):
    matches = _get_ckpts_sorted(f'{prefix}*/{C.CKPT_NAME_LAST}')
    assert len(matches) == 1
    return matches[0]


def load_conf(conf_name, print_conf=False):
    # load conf from ckpts/conf_name/conf.yaml
    conf_path = Path(C.CKPT_ROOT) / conf_name / C.CONF_NAME
    if not conf_path.exists():
        raise FileNotFoundError(f"Config file not found at {conf_path}")
    conf = OmegaConf.load(conf_path)
    if print_conf:
        print(OmegaConf.to_yaml(conf))
    return conf


@func_rank_0
def save_conf(conf, print_conf=False):
    # save conf as ckpts/conf.name/conf.yaml
    conf_yaml = OmegaConf.to_yaml(conf)
    conf_path = Path(C.CKPT_ROOT) / str(conf.name) / C.CONF_NAME
    conf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(conf_path, 'w') as f:
        f.write(conf_yaml)
    if print_conf:
        print(conf_yaml)
