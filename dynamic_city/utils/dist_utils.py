import torch.distributed as dist
import wandb


def distributed():
    return dist.is_initialized()


def rank_0():
    return not distributed() or dist.get_rank() == 0


def func_rank_0(func):
    return lambda *args, **kwargs: func(*args, **kwargs) if rank_0() else None


@func_rank_0
def write_text(text):
    print(text, end='')


@func_rank_0
def flush_text():
    print('\r', end='')


@func_rank_0
def print_text(*args, **kwargs):
    print(*args, **kwargs)


@func_rank_0
def wlog(*args, **kwargs):
    wandb.log(*args, **kwargs)
