from .yacs import  CfgNode as CN
import argparse
import os
import numpy as np
from . import yacs

# root config object
cfg = CN()

cfg.scene = 'default'

# task
cfg.task = "default"
cfg.task_args = CN()

# experiment name
cfg.exp_name = "default"

# pretrain
cfg.pretrain = ''

# gpu_ids
cfg.gpu_ids = [0]

# datasets
cfg.train_dataset_module = ""
cfg.val_dataset_module = ""
cfg.test_dataset_module = ""

# network
cfg.network_module = ""

# loss
cfg.loss_module = ""

# 分布式训练
cfg.distributed = False

# if load the pretrained model
cfg.resume = True

# epoch
cfg.epoch_iter = -1
cfg.save_epoch = 100000
cfg.save_latest_epoch = 1
cfg.eval_epoch = 1

# log
cfg.log_interval = 20

# trained model
cfg.trained_model_dir = 'data/trained_model'

# recorder
cfg.recorder_dir = 'data/recorder'

# result
cfg.result_dir = 'data/result'

# evaluator
cfg.evaluator_module = ""
cfg.skip_eval = False

# visualizer
cfg.visualizer_module = ""

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()
cfg.train.num_epoch = 10000
cfg.train.num_workers = 4
cfg.train.shuffle = True
cfg.train.eps = 1e-8
cfg.train.optimizer = "adam"
cfg.train.learning_rate = 1e-3
cfg.train.weight_decay = 0.
cfg.train.batch_size = 4
cfg.train.collator = 'default'
cfg.train.batch_sampler = 'default'
cfg.train.sampler_meta = CN({})
cfg.train.lr_scheduler = CN({'type': 'multi_step', 'milestones': [80, 120, 200, 240], 'gamma': 0.5})

# -----------------------------------------------------------------------------
# test
# -----------------------------------------------------------------------------
cfg.test = CN()
cfg.test.num_epoch = -1
cfg.test.batch_size = 1
cfg.test.collator = 'default'
cfg.test.batch_sampler = 'default'
cfg.test.sampler_meta = CN({})


def parse_cfg(cfg, args):
    """
    parse the config object
    """
    # task must be specified
    if len(cfg.task) == 0:
        raise ValueError("task is not specified")

    # assign the gpus (-1 means cpu)
    if -1 not in cfg.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in cfg.gpu_ids])
    cfg.local_rank = args.local_rank # The gpu id where the current process is located



    # experiment name
    print('{}: {}'.format(cfg.task, cfg.exp_name))

def make_config(args):
    """
    make config object from the config file(or merge a config file) and the args
    """
    def merge_cfg(cfg_file, cfg):
        with open(cfg_file, "r") as f:
            current_cfg = yacs.load_cfg(f)
        if 'parent_cfg' in current_cfg.keys():
            cfg = merge_cfg(current_cfg.parent_cfg, cfg)
            cfg.merge_from_other_cfg(current_cfg)
        else:
            cfg.merge_from_other_cfg(current_cfg)
        print(cfg_file)
        return cfg
    cfg_ = merge_cfg(args.cfg_file, cfg)
    # try:
    #     index = args.opts.index('other_opts')
    #     cfg_.merge_from_list(args.opts[:index])
    # except:
    #     cfg_.merge_from_list(args.opts)
    parse_cfg(cfg_, args)
    return cfg_


parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="configs/default.yaml", type=str)
parser.add_argument("--test", action="store_true", dest="test", default=False)
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()
cfg = make_config(args)