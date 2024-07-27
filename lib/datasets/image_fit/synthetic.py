import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()

    def __getitem__(self, item):
        return item

    def __len__(self):
        return 1