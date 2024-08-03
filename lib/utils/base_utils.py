import torch
from torch import nn

def to_cuda(batch, device=torch.device('cuda:0')):
    """
    Move batch to cuda device
    """
    if isinstance(batch, tuple) or isinstance(batch, list): # tuple or list
        batch = [to_cuda(b, device) for b in batch]
    elif isinstance(batch, dict): # dict
        batch_ = {}
        for key in batch:
            if key == 'meta':
                batch_[key] = batch[key]
            else:
                batch_[key] = to_cuda(batch[key], device)
        batch = batch_
    else: # tensor or others
        batch = batch.to(device)
    return batch