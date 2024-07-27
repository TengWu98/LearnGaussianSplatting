import importlib

from .trainer import Trainer

def _wrapper_factory(cfg, network, train_loader=None):
    module_name = cfg.loss_module
    module = importlib.import_module(module_name)
    network_wrapper_class = getattr(module, 'NetworkWrapper')
    network_wrapper = network_wrapper_class(network, train_loader)
    return network_wrapper

def make_trainer(cfg, network, train_loader=None):
    network = _wrapper_factory(cfg, network, train_loader)
    return Trainer(network)