from collections import Counter

from lib.train.optimizers.lr_schedulers import MultiStepLR, ExponentialLR

def make_lr_scheduler(cfg, optimizer):
    cfg_lr_scheduler = cfg.train.lr_scheduler
    if cfg_lr_scheduler.type == 'multi_step':
        lr_scheduler = MultiStepLR(optimizer, milestones=cfg_lr_scheduler.milestones, gamma=cfg_lr_scheduler.gamma)
    elif cfg_lr_scheduler.type == 'ExponentialLR':
        lr_scheduler = ExponentialLR(optimizer, decay_epochs=cfg_lr_scheduler.decay_epochs, gamma=cfg_lr_scheduler.gamma)
    return lr_scheduler

def set_lr_scheduler(cfg, lr_scheduler):
    cfg_lr_scheduler = cfg.train.lr_scheduler
    if cfg_lr_scheduler.type == 'multi_step':
        lr_scheduler.milestones = Counter(cfg_lr_scheduler.milestones)
    elif cfg_lr_scheduler.type == 'ExponentialLR':
        lr_scheduler.decay_epochs = cfg_lr_scheduler.decay_epochs
    lr_scheduler.gamma = cfg_lr_scheduler.gamma