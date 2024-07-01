import torch 
from torch.utils.data import Dataset
from pathlib import Path 
from cv_utils import (prepare_styletransfer_model,
                      inference_styletransfer, 
                      render_batch,
                      view_ST_outputs)

from utils import device, read_mask
import numpy as np 
import cv2 as cv

class STDataset(Dataset):

    def __init__(self, config, mesh, poses):
        self.config = config # already read in
        assert isinstance(poses, list), 'poses should be a list of SE(3)'
        self.poses = poses 
        self.I2I_trainer = prepare_styletransfer_model(config)
        self.mesh = mesh 
        self.mask = read_mask(config)
        self.mask = cv.resize(self.mask, (320, 320))
        self.mask = np.stack((self.mask, self.mask, self.mask), axis=2)

    def __len__(self):
        return len(self.poses)
    
    def __getitem__(self, idx):
        # print('i got called')
        pose = self.poses[idx]
        positions_batch = torch.Tensor(pose[:3,3]).to(device)
        R_batch = torch.Tensor(pose[:3,:3]).to(device)
        positions_batch, R_batch = positions_batch[None], R_batch[None]
        render_rgb, _ = render_batch(self.config, self.mesh, R_batch, positions_batch, 1, device=device)
        ST_image = inference_styletransfer(self.I2I_trainer, render_rgb, self.mask)
        # positions_batch = torch.squeeze(positions_batch)
        # R_batch = torch.squeeze(R_batch)
        
        # view_ST_outputs(render_rgb, ST_image)
        return positions_batch, R_batch, ST_image, render_rgb
