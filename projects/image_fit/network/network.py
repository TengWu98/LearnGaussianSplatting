import torch
import torch.nn as nn

from lib.config import cfg

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()