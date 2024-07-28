import importlib
from random import shuffle

from torch.utils.data import DataLoader

def make_dataset(cfg, is_train=True):
    """
    :param cfg:
    :return:
    """
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

def make_data_loader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.train.batch_size
        shuffle = cfg.train.shuffle
        drop_last = False
    else:
        batch_size = cfg.test.batch_size
        shuffle = False
        drop_last = False

    dataset = make_dataset(cfg, is_train)
    num_workers = cfg.train.num_workers

    data_loader = DataLoader(dataset,
                             num_workers=num_workers)

    return data_loader