import torch
import os

from lib.config import cfg, args
from lib.networks import make_network
from lib.datasets import make_data_loader
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, set_lr_scheduler, make_recorder
from lib.evaluators import make_evaluator


def train(cfg, network):
    # data loader
    train_loader = make_data_loader(cfg, is_train=True, is_distributed=cfg.distributed, max_iter=cfg.epoch_iter)
    val_loader = make_data_loader(cfg, is_train=False)

    # trainer
    trainer = make_trainer(cfg, network, train_loader)
    optimizer = make_optimizer(cfg, network)
    lr_scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)
    evaluator = make_evaluator(cfg)


    set_lr_scheduler(cfg, lr_scheduler)

    return network

def test(cfg, network):
    # data loader
    val_loader = make_data_loader(cfg, is_train=False)

    # trainer
    trainer = make_trainer(cfg, network)

    # evaluator
    evaluator = make_evaluator(cfg)


def main():
    if cfg.distributed:
        # TODO(WT) 处理分布式训练逻辑
        pass

    network = make_network(cfg)

    if args.test:
        test(cfg, network)
    else:
        train(cfg, network)

if __name__ == '__main__':
    main()

