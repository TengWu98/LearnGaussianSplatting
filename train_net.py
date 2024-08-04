import torch
import os

from lib.config import cfg, args
from lib.networks import make_network
from lib.datasets import make_data_loader
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, set_lr_scheduler, make_recorder
from lib.evaluators import make_evaluator
from lib.utils.base_utils import is_main_process
from lib.utils.net_utils import load_model, save_model, load_pretrain, load_network


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

    # load model/pretrainW
    begin_epoch = load_model(network, optimizer, recorder, lr_scheduler, cfg.trained_model_dir, resume=cfg.resume)
    if begin_epoch == 0 and cfg.pretrain != '':
        load_pretrain(network, cfg.pretrain)

    # set lr_scheduler
    set_lr_scheduler(cfg, lr_scheduler)

    # training loop
    for epoch in range(begin_epoch, cfg.train.num_epoch):
        recorder.epoch = epoch
        if cfg.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)
        train_loader.dataset.epoch = epoch

        trainer.train(epoch, train_loader, optimizer, recorder)
        lr_scheduler.step()

        # save model
        if (epoch + 1) % cfg.save_epoch == 0 and is_main_process():
            save_model(network, optimizer, recorder, lr_scheduler, cfg.trained_model_dir, epoch)
        if (epoch + 1) % cfg.save_latest_epoch == 0 and is_main_process():
            save_model(network, optimizer, recorder, lr_scheduler, cfg.trained_model_dir, epoch, last=True)
        if (epoch + 1) % cfg.eval_epoch == 0 and is_main_process():
            trainer.val(epoch, val_loader, evaluator)

    return network

def test(cfg, network):
    val_loader = make_data_loader(cfg, is_train=False)
    trainer = make_trainer(cfg, network)
    evaluator = make_evaluator(cfg)

    # load model and test
    epoch = load_network(network, cfg.trained_model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    trainer.val(epoch, val_loader, evaluator)


def main():
    if cfg.distributed:
        # TODO(WT) 处理分布式训练逻辑
        pass

    network = make_network(cfg)

    if args.test:
        test(cfg, network)
    else:
        train(cfg, network)

    if is_main_process():
        print('Success!')
        print('=' * 80)
    os.system('kill -9 {}'.format(os.getpid()))

if __name__ == '__main__':
    main()

