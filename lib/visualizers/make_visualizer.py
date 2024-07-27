import importlib

def make_visualizer(cfg):
    """
    :param cfg:
    :return:
    """
    module_name = cfg.visualizer_module
    module = importlib.import_module(module_name)
    visualizer_class = getattr(module, 'Visualizer')
    return visualizer_class()