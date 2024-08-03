import torch

from lib.train.optimizers.radam import RAdam

_optimizer_factory = {
    'adam': torch.optim.Adam,
    'radam': RAdam,
    'sgd': torch.optim.SGD
}

def make_optimizer(cfg, net):
    params = []
    lr = cfg.train.learning_rate
    weight_decay = cfg.train.weight_decay
    eps = cfg.train.eps

    for key, value in net.named_parameters():
        if not value.requires_grad:
            continue
        params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]

    if 'adam' in cfg.train.optimizer:
        optimizer = _optimizer_factory[cfg.train.optimizer](params, lr, weight_decay=weight_decay, eps=eps)
    else:
        optimizer = _optimizer_factory[cfg.train.optimizer](params, lr, momentun=0.9)

    return optimizer