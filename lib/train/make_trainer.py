import importlib
import torch
import torch.nn as nn
import tqdm
import time
import datetime

from lib.config import cfg
from lib.utils.base_utils import to_cuda, is_main_process

class Trainer(object):
    def __init__(self, network):
        device = torch.device('cuda:{}'.format(cfg.local_rank))
        network = network.to(device)

        # distributed training
        # TODO(WT)
        if cfg.distributed:
            pass

        self.network = network
        self.local_rank = cfg.local_rank
        self.device = device
        self.global_step = 0

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def train(self, epoch, data_loader, optimizer, recorder):
        max_iter = len(data_loader)
        self.network.train()
        end = time.time()
        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end   # measure data loading time
            iteration = iteration + 1       # 1-based index

            # forward pass
            batch = to_cuda(batch, self.device)
            batch['step'] = self.global_step
            output, loss, loss_stats, image_stats = self.network(batch)

            # backward pass
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(self.network.parameters(), 40) # clip gradient
            optimizer.step()

            if not is_main_process():
                continue

            # data recording
            recorder.step += 1

            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            self.global_step += 1

            # periodically print training stats.
            if iteration % cfg.log_interval == 0 or iteration == max_iter:
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration + 1)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)

                recorder.update_image_stats(image_stats)
                recorder.record('train')


    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        image_stats = {}
        data_size = len(data_loader)

        for batch in tqdm.tqdm(data_loader):
            batch = to_cuda(batch, self.device)
            batch['step'] = recorder.step if recorder is not None else 0
            with torch.no_grad():
                output, loss, loss_stats, _ = self.network(batch)
                if evaluator is not None:
                    image_stats_ = evaluator.evaluate(output, batch)
                    if image_stats_ is not None:
                        image_stats.update(image_stats_)

            loss_stats = self.reduce_loss_stats(loss_stats)
            for k, v in loss_stats.items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v

        loss_state = []
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        print(loss_state)

        if evaluator is not None:
            result = evaluator.summarize()
            val_loss_stats.update(result)

        if recorder:
            recorder.record('val', epoch, val_loss_stats, image_stats)

def _wrapper_factory(cfg, network, train_loader=None):
    module_name = cfg.loss_module
    module = importlib.import_module(module_name)
    network_wrapper_class = getattr(module, 'NetworkWrapper')
    network_wrapper = network_wrapper_class(network, train_loader)
    return network_wrapper

def make_trainer(cfg, network, train_loader=None):
    network = _wrapper_factory(cfg, network, train_loader)
    return Trainer(network)