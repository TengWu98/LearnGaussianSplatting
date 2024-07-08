from .yacs import  CfgNode as CN
import argparse
import os
import numpy as np
from . import yacs

# create a config object
cfg = CN()

# experiment name
cfg.experiment_name = "default"

# gpus
cfg.gpus = [0]

# epoch


def parse_cfg(cfg, args):
    """
    parse the config object
    :param cfg: cfg object
    :param args:
    :return: None
    """
    # assign the gpus (-1 means cpu)
    if -1 not in cfg.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in cfg.gpus])

    # experiment name
    print('Experiment name: {}'.format(cfg.experiment_name))

def make_config(args):
    """
    make config object from the config file(or merge a config file) and the args
    :param args:
    :return: cfg object
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
args = parser.parse_args()
cfg = make_config(args)