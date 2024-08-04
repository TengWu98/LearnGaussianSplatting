import torch
import torch.nn as nn

class NetworkWrapper(nn.Module):
    def __init__(self, network, train_loader):
        super(NetworkWrapper, self).__init__()
        self.network = network
        self.color_crit = nn.MSELoss(reduction='mean')
        self.mse2pnsr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.])) # convert mse to psnr

    def forward(self, batch):
        output = self.network(batch)

        scalar_stats = {}
        loss = 0
        color_loss = self.color_crit(output['rgb'], batch['rgb'])
        scalar_stats.update({'color_mse': color_loss})
        loss += color_loss

        psnr = -10. * torch.log(color_loss.detach()) / torch.log(torch.Tensor([10.]).to(color_loss.device))
        scalar_stats.update({'psnr': psnr})

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats