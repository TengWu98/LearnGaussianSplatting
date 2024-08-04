import torch
from lib.networks.encoding.freq import Encoder as FreqEncoder
from lib.config import cfg as global_cfg

def get_encoder(cfg):
    if cfg.type == 'frequency':
        encoder_kwargs = {
                'include_input' : True,
                'input_dims' : cfg.input_dim,
                'max_freq_log2' : cfg.freq - 1,
                'num_freqs' : cfg.freq,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
        }
        encoder_obj = FreqEncoder(**encoder_kwargs)
        encoder = lambda x, eo=encoder_obj: eo.embed(x)
        return encoder, encoder_obj.out_dim
    else:
        raise NotImplementedError
