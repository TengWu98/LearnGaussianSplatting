import torch
import os
from torch import nn
import numpy as np
from termcolor import colored

from lib.config import cfg

def save_model(network, optimizer, lr_scheduler, recorder, model_dir, epoch, last=False):
    """
    Save model to the latest checkpoint.
    """
    os.system('mkdir -p {}'.format(model_dir))
    model = {
        'network': network.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'recorder': recorder.state_dict(),
        'epoch': epoch
    }

    if last:
        torch.save(model, os.path.join(model_dir, 'latest.pth'))
    else:
        torch.save(model, os.path.join(model_dir, '{}.pth'.format(epoch)))

    # remove previous pretrained model if the number of models is too big
    pths = [
        int(pth.split('.')[0]) for pth in os.listdir(model_dir)
        if pth != 'latest.pth'
    ]
    if len(pths) <= 5:
        return
    os.system('rm {}'.format(
        os.path.join(model_dir, '{}.pth'.format(min(pths)))))

def load_model(network, optimizer, lr_scheduler, recorder, model_dir, resume=True, epoch=-1):
    """
    Load model from the latest checkpoint.
    """
    if not resume:
        os.system('rm -rf {}'.format(model_dir))

    if not os.path.exists(model_dir):
        return 0

    pths = [
        int(pth.split('.')[0]) for pth in os.listdir(model_dir)
        if pth != 'latest.pth'
    ]
    if len(pths) == 0 and 'latest.pth' not in os.listdir(model_dir):
        return 0

    if epoch == -1:
        if 'latest.pth' in os.listdir(model_dir):
            pth = 'latest'
        else:
            pth = max(pths)
    else:
        pth = epoch
    print('load model: {}'.format(os.path.join(model_dir,
                                               '{}.pth'.format(pth))))

    pretrained_model = torch.load(
        os.path.join(model_dir, '{}.pth'.format(pth)), 'cpu')
    network.load_state_dict(pretrained_model['network'])
    if 'optimizer' in pretrained_model:
        optimizer.load_state_dict(pretrained_model['optimizer'])
        lr_scheduler.load_state_dict(pretrained_model['lr_scheduler'])
        recorder.load_state_dict(pretrained_model['recorder'])
        return pretrained_model['epoch'] + 1
    else:
        return 0

def load_network(network, model_dir, resume=True, epoch=-1, strict=True):
    if not resume:
        return 0

    if not os.path.exists(model_dir):
        print(colored('pretrained model does not exist', 'red'))
        return 0

    if os.path.isdir(model_dir):
        pths = [
            int(pth.split('.')[0]) for pth in os.listdir(model_dir)
            if pth != 'latest.pth'
        ]
        if len(pths) == 0 and 'latest.pth' not in os.listdir(model_dir):
            return 0

        if epoch == -1:
            if 'latest.pth' in os.listdir(model_dir):
                pth = 'latest'
            else:
                pth = max(pths)
        else:
            pth = epoch

        model_path = os.path.join(model_dir, '{}.pth'.format(pth))
    else:
        model_path = model_dir

    print('load model: {}'.format(model_path))
    pretrained_model = torch.load(model_path)
    network.load_state_dict(pretrained_model['network'], strict=strict)
    if 'epoch' in pretrained_model:
        return pretrained_model['epoch'] + 1
    else:
        return 0

def load_pretrain(network, model_dir):
    """
    Load pretrained model.
    """
    pass

def save_pretrain(network, task, model_dir):
    """
    Save pretrained model.
    """
    pass
