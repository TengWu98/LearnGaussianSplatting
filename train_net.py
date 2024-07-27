import torch
import os

from lib.config import cfg, args
from lib.networks import make_network
from lib.datasets import make_data_loader
from lib.train import make_trainer
from lib.evaluators import make_evaluator


def train(cfg, network):
    # data loader
    train_loader = make_data_loader(cfg, is_train=True)
    val_loader = make_data_loader(cfg, is_train=False)

    # trainer
    trainer = make_trainer(cfg, network, train_loader)

    # evaluator
    evaluator = make_evaluator(cfg)

    return network

def test(cfg, network):
    # data loader
    val_loader = make_data_loader(cfg, is_train=False)

    # trainer
    trainer = make_trainer(cfg, network)

    # evaluator
    evaluator = make_evaluator(cfg)


def main():
    network = make_network(cfg)

    if args.test:
        test(cfg, network)
    else:
        train(cfg, network)

if __name__ == '__main__':
    main()

