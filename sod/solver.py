import torch
from dfi import build_model
import numpy as np
import os
import cv2

import warnings

warnings.filterwarnings('ignore')

class Solver(object):
    def __init__(self, data_loader, model_path, config=0):
        self.data_loader = data_loader
        self.config = config
        self.net = build_model()
        # if self.config.cuda:
        #     self.net = self.net.cuda()
        print(f'Loading pre-trained model from {model_path}...')
        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()

    def test(self):
        for _, data_batch in enumerate(self.data_loader):
            images, name = data_batch['image'], data_batch['name'][0]
            with torch.no_grad():
                # if self.config.cuda: images = images.cuda()
                preds = self.net(images, mode=3)
                pred_sal = np.squeeze(torch.sigmoid(preds[1][0]).cpu().data.numpy())
                cv2.imwrite(os.path.join(self.config.test_fold, name[:-4] + '_sal.png'), 255 * pred_sal)

        print("Testing Finished.")