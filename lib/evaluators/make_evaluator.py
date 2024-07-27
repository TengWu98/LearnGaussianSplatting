import importlib

def _evaluator_factory(cfg):
    module_name = cfg.evaluator_module
    module = importlib.import_module(module_name)
    evaluator_class = getattr(module, 'Evaluator')
    return evaluator_class()

def make_evaluator(cfg):
    """
    :param cfg:
    :return:
    """
    if cfg.skip_eval:
        return None
    return _evaluator_factory(cfg)