import numpy as np
import torch
import torch.nn.functional as F

import dynamic_city.utils.constants as C
from dynamic_city.utils.data_utils import parse_semantic_dict
from dynamic_city.utils.lovasz import lovasz_softmax


def _kl_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def _get_ce_function(ce_weight=None, **kwargs):
    """
    pred: batch_size, num_classes, ...
    gt: batch_size, ...,
    """
    return lambda pred, gt: F.cross_entropy(pred, gt.long(), weight=ce_weight)


def _get_lovasz_function(**kwargs):
    """
    pred: -1, num_classes
    gt: -1,
    """
    return lambda pred, gt: lovasz_softmax(F.softmax(pred, dim=-1), gt)


def _get_kl_function(**kwargs):
    """
    mus: [mu] * 6
    logvars: [logvar] * 6
    """
    return lambda mus, logvars: (sum([_kl_loss(mu, logvar) for mu, logvar in zip(mus, logvars)]) /
                                 sum([np.prod(list(mu.shape)) for mu in mus]))


_loss_functions = {
    'ce': _get_ce_function,
    'lovasz': _get_lovasz_function,
    'kl': _get_kl_function,
}


def calculate_ce_weight(dataset_conf):
    assert hasattr(dataset_conf, 'semantic_map') and hasattr(dataset_conf, 'semantic_frequency')
    semantic_map = parse_semantic_dict(dataset_conf.semantic_map)
    semantic_frequency = parse_semantic_dict(dataset_conf.semantic_frequency)
    assert len(semantic_map) == len(semantic_frequency)
    mapped_frequency = np.zeros(dataset_conf.num_classes, dtype=int)
    np.add.at(mapped_frequency, semantic_map, semantic_frequency)
    mapped_percentage = mapped_frequency / (np.sum(mapped_frequency) + C.EPSILON)
    ce_weight = np.power(np.amax(mapped_percentage) / (mapped_percentage + C.EPSILON), 1 / 3.0)
    ce_weight = torch.tensor(ce_weight)
    return ce_weight


def build_losses(model_conf, **kwargs):
    losses = dict()
    for loss_name in model_conf.loss_weights.keys():
        losses[loss_name] = _loss_functions[loss_name](**kwargs)
    return losses
