import torch.utils.data as data
import os
import json
import imageio.v2 as imageio
import numpy as np
import cv2

from lib.config import cfg

class Dataset(data.Dataset):
    """
    image_fit Dataset: only fit one image: lego view 0
    """
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        data_root, split, scene = kwargs['data_root'], kwargs['split'], cfg.scene
        view = kwargs['view']
        self.input_ratio = kwargs['input_ratio']
        self.scene_data_root = os.path.join(data_root, scene)
        self.split = split
        self.batch_size = cfg.task_args.N_pixels

        # read images
        image_paths = []
        json_info = json.load(open(os.path.join(self.scene_data_root, 'transforms_train.json')))
        for frame in json_info['frames']:
            image_paths.append(os.path.join(self.scene_data_root, frame['file_path'][2:] + '.png'))

        img = imageio.imread(image_paths[view]) / 255.
        img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])  # alpha blend RGB image with white background
        if self.input_ratio != 1.:
            img = cv2.resize(img, None, fx=self.input_ratio, fy=self.input_ratio, interpolation=cv2.INTER_AREA)

        # set image
        self.img = np.array(img).astype(np.float32)

        # set uv
        H, W = img.shape[:2]
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        u, v = X.astype(np.float32) / (W - 1), Y.astype(np.float32) / (H - 1)
        self.uv = np.stack([u, v], -1).reshape(-1, 2).astype(np.float32)

    def __getitem__(self, item):
        if self.split == 'train':
            ids = np.random.choice(len(self.uv), self.batch_size, replace=False)
            uv = self.uv[ids]
            rgb = self.img.reshape(-1, 3)[ids]
        else:
            uv = self.uv
            rgb = self.img.reshape(-1, 3)

        ret = {'uv': uv, 'rgb': rgb}
        ret.update({'meta': {'H': self.img.shape[0], 'W': self.img.shape[1]}})
        return ret

    def __len__(self):
        return 1