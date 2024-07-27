import importlib

def make_network(cfg):
    """
    :param cfg:
    :return:
    """
    module_name = cfg.network_module
    module = importlib.import_module(module_name)
    network_class = getattr(module, 'Network')
    return network_class()