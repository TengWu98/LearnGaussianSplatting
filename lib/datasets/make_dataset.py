import importlib
from random import shuffle
import time
import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate

from . import samplers

from torch.utils.data import DataLoader

def make_dataset(cfg, is_train=True):
    if is_train:
        args = cfg.train_dataset
        module_name = cfg.train_dataset_module
    else:
        args = cfg.test_dataset
        module_name = cfg.test_dataset_module

    module = importlib.import_module(module_name)
    dataset_class = getattr(module, 'Dataset')
    dataset = dataset_class(**args)
    return dataset

def make_collator(cfg, is_train):
    """
    TODO(WT) 提供可配置的collator
    """
    return default_collate


def make_data_sampler(dataset, shuffle, distributed):
    """
    """
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    return sampler


def make_data_batch_sampler(cfg, smapler, batch_size, drop_last, max_iter, is_train):
    """
    :param drop_last: when the size of the last batch is less than batch_size, drop it if drop_last is True
    :param max_iter: max number of iterations. If set to -1, there is no limit on the number of iterations.
    """
    if is_train:
        batch_sampler = cfg.train.batch_sampler
        sampler_meta = cfg.train.sampler_meta
    else:
        batch_sampler = cfg.test.batch_sampler
        sampler_meta = cfg.test.sampler_meta

    if batch_sampler == "default":
        batch_sampler = torch.utils.data.BatchSampler(smapler, batch_size, drop_last)
    elif batch_sampler == "image_size":
        batch_sampler = samplers.ImageSizeBatchSampler(smapler, batch_size, drop_last, sampler_meta)

    if max_iter != -1:
        batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, max_iter)
    return batch_sampler

def worker_init_fn(worker_id):
    """
    When using multi-thread data loading, ensure that each worker thread has a different random number generator seed
    to avoid generating duplicate random number sequences. The range is [0, 65535].
    """
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2**16))))

def make_data_loader(cfg, is_train=True, is_distributed=False, max_iter=-1):
    """
    TODO(WT) 补充注释
    """
    if is_train:
        batch_size = cfg.train.batch_size
        shuffle = cfg.train.shuffle
        drop_last = False
    else:
        batch_size = cfg.test.batch_size
        shuffle = True if is_distributed else False
        drop_last = False

    dataset = make_dataset(cfg, is_train)
    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_data_batch_sampler(cfg, sampler, batch_size, drop_last, max_iter, is_train)
    collator = make_collator(cfg, is_train)
    num_workers = cfg.train.num_workers

    data_loader = DataLoader(dataset,
                             batch_sampler=batch_sampler,
                             collate_fn=collator,
                             num_workers=num_workers,
                             worker_init_fn=worker_init_fn,
                             pin_memory=True)

    return data_loader