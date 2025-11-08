import os, sys
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from ..model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from .IFNet_HDv3 import *
import torch.nn.functional as F
from ..model.loss import *
sys.path.append('../../')
from deforum.utils.general import checksum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model:
    def __init__(self, local_rank=-1):
        self.flownet = IFNet()
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-4)
        self.epe = EPE()
        self.version = 3.9
        # self.vgg = VGGPerceptualLoss().to(device)
        self.sobel = SOBEL()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)
         
    def load_model(self, path, rank, deforum_models_path):
        
        download_rife_model(path, deforum_models_path)

        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param
        if rank <= 0:
            if torch.cuda.is_available():
                self.flownet.load_state_dict(convert(torch.load(os.path.join(deforum_models_path,'{}.pkl').format(path))), False)
            else:
                self.flownet.load_state_dict(convert(torch.load(os.path.join(deforum_models_path,'{}.pkl').format(path), map_location ='cpu')), False)

    def inference(self, img0, img1, timestep=0.5, scale=1.0):
        # Get dimensions (batch, channels, height, width)
        _, _, ph, pw = img0.shape

        # Create warp grid tensors for optical flow
        tenFlow_div = torch.tensor(
            [(pw - 1.0) / 2.0, (ph - 1.0) / 2.0],
            dtype=torch.float,
            device=img0.device
        )

        tenHorizontal = torch.linspace(-1.0, 1.0, pw, dtype=torch.float, device=img0.device)
        tenHorizontal = tenHorizontal.view(1, 1, 1, pw).expand(-1, -1, ph, -1)
        tenVertical = torch.linspace(-1.0, 1.0, ph, dtype=torch.float, device=img0.device)
        tenVertical = tenVertical.view(1, 1, ph, 1).expand(-1, -1, -1, pw)
        backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1)

        # Encode features using Head encoder (v4.26 requirement)
        f0 = self.flownet.encode(img0)
        f1 = self.flownet.encode(img1)

        # Create timestep tensor (shape: [batch, 1, height, width])
        timestep_tensor = torch.full(
            [img0.shape[0], 1, ph, pw],
            timestep,
            dtype=torch.float,
            device=img0.device
        )

        # Call flownet with v4.26 signature
        # Returns interpolated frame directly (not a list)
        output = self.flownet(img0, img1, timestep_tensor, tenFlow_div, backwarp_tenGrid, f0, f1)

        return output
    
    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        scale = [8, 4, 2, 1]
        flow, mask, merged = self.flownet(torch.cat((imgs, gt), 1), scale=scale, training=training)
        loss_l1 = (merged[3] - gt).abs().mean()
        loss_smooth = self.sobel(flow[3], flow[3]*0).mean()
        # loss_vgg = self.vgg(merged[2], gt)
        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_cons + loss_smooth * 0.1
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow[2]
        return merged[3], {
            'mask': mask,
            'flow': flow[3][:, :2],
            'loss_l1': loss_l1,
            'loss_cons': loss_cons,
            'loss_smooth': loss_smooth,
            }

def download_rife_model(path, deforum_models_path):
    # RIFE v4.26 from official Practical-RIFE repository (2024.09.21)
    # Compatible with IFNet_HDv3 v4.26 architecture (5-block, Head encoder, feature propagation)
    # Google Drive file ID: 1gViYvvQrtETBgU1w8axZSsr7YUuw31uy
    import hashlib
    import zipfile

    options = {'RIFE426': (
               '2c18c32ad467acd16a773c7068be25f0c0eb84f85f16719d72b2046617f22fe2fe441b1b5e16c8419e1a396c7ed6e6227901bfd7a1419475e225d02435bae681',
               '1gViYvvQrtETBgU1w8axZSsr7YUuw31uy')}
    if path in options:
        target_file = f"{path}.pkl"
        target_path = os.path.join(deforum_models_path, target_file)
        if not os.path.exists(target_path):
            import gdown
            print(f"Downloading RIFE model {path} from Google Drive...")

            # Download ZIP archive
            temp_zip_path = os.path.join(deforum_models_path, f"{path}_temp.zip")
            gdown.download(id=options[path][1], output=temp_zip_path, quiet=False)

            # Extract flownet.pkl from train_log/flownet.pkl
            print(f"Extracting RIFE model from archive...")
            try:
                with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                    # Extract the flownet.pkl file
                    zip_ref.extract('train_log/flownet.pkl', deforum_models_path)

                # Move extracted file to target location
                extracted_path = os.path.join(deforum_models_path, 'train_log/flownet.pkl')
                os.rename(extracted_path, target_path)

                # Clean up extracted directory structure
                import shutil
                extracted_dir = os.path.join(deforum_models_path, 'train_log')
                if os.path.exists(extracted_dir):
                    shutil.rmtree(extracted_dir)

                # Verify checksum of extracted file
                if checksum(target_path, hashlib.sha512) != options[path][0]:
                    os.remove(target_path)
                    raise Exception(f"Checksum mismatch for {target_file}. Please download manually from: https://drive.google.com/file/d/{options[path][1]}/view and place in: " + deforum_models_path)

                print(f"Successfully downloaded and extracted RIFE model to {target_path}")
            finally:
                # Clean up temporary ZIP file
                if os.path.exists(temp_zip_path):
                    os.remove(temp_zip_path)
