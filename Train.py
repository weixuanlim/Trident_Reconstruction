import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchscale.architecture.config import EncoderConfig
from torchscale.architecture.encoder import Encoder
from functools import partial
import numpy as np
import pandas as pd
import os
import json
import re
from fastai.vision.all import *
from fastai.vision.all import OptimWrapper
import fastai
import random

"""
Dataset Preparation

for every event, return x: two dictionary {x0,y0,z0,r0,charge}, {target}

the input feature is retrieved from preprocessed dataframe stored in feather file
(column: x0, y0, t0 time, charge)
the label is stored in theb original mc_events.json file
"""
def get_direction(x,y,z):
    azimuth = torch.atan2(y, x)
    azimuth = torch.where(azimuth < 0, azimuth + 2 * torch.pi, azimuth)
    r = torch.sqrt(x**2 + y**2 + z**2)
    zenith = torch.acos(z / r)
    return torch.stack((azimuth, zenith),dim=-1)

def pad_dataframe(df, target_length): #padding for fixed input, this dataframe will
    L = len(df)
    # Calculate the number of rows to add
    num_rows_to_add = target_length - L
    # Create a DataFrame with zeros for padding
    padding_df = pd.DataFrame(np.zeros((num_rows_to_add, len(df.columns))), columns=df.columns)
    padded_df = pd.concat([df, padding_df], ignore_index=True)
    return padded_df, L
       
class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, mode):
        self.root_dir = root_dir
        self.mode = mode
        # Getting file from directory
        all_entries = os.listdir(root_dir)
        batch_folders = [f for f in all_entries if re.match(r'newbatch\d+', f)]
        self.folder_paths = sorted(batch_folders, key=lambda x: int(re.search(r'\d+', x).group()))
        if mode == "train":
            self.folder_paths = self.folder_paths[:220]  # Folders 1-220 for training
            
        elif mode == "val":
            self.folder_paths = self.folder_paths[220:230]  # Folders 221-230 for validation

        self.folder_paths = [os.path.join(root_dir, f) for f in self.folder_paths]
        self.target_cache = {}  # Cache to store loaded JSON target data

    def __len__(self):
        return len(self.folder_paths) * 1000  # Total number of data files

    def __getitem__(self, idx):
        folder_idx = idx // 1000
        file_idx = idx % 1000
        folder_path = self.folder_paths[folder_idx]

        # Load data from feather file
        data_path = os.path.join(folder_path, f"event_{file_idx}.feather")
        data = pd.read_feather(data_path)
        
        data, L = pad_dataframe(data, 196)
        attn_mask = torch.zeros(196, dtype=torch.bool)
        attn_mask[:L] = True
        data_dict = {"x0": torch.from_numpy(data["x0"].values).float(),
            "y0": torch.from_numpy(data["y0"].values).float(),
            "z0": torch.from_numpy(data["z0"].values).float(),
            "t0": torch.from_numpy(data["t0"].values).float(),
            "charge": torch.from_numpy(data["charge"].values).float(),
            "mask": attn_mask}

        if folder_path not in self.target_cache:
            target_path = os.path.join(folder_path, "mc_events.json")
            with open(target_path, 'r') as jfile:
                self.target_cache[folder_path] = json.load(jfile)
        
        target_data = self.target_cache[folder_path]
        info = target_data[file_idx]['particles_in'][0]
        target_value = get_direction(torch.tensor(info['px']),torch.tensor(info['py']),torch.tensor(info['pz'])) 
        target_dict = {"target": target_value.float()}
        
        return data_dict, target_dict
"""
Some emdedding are done before projecting the data into encoder block
"""
class SinusoidalEncoding(nn.Module):
    def __init__(self, dim, M=10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[..., None] * emb[None, ...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class InputProjection(nn.Module):
    def __init__(self, seq_length, hidden_features, out_features):
        super().__init__()
        self.emb = SinusoidalEncoding(dim=seq_length)
 
        self.mlp = nn.Sequential(nn.Linear(5*seq_length, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features))

    def forward(self, x):
        x0, y0, z0, t0, charge = x["x0"], x["y0"], x["z0"], x["t0"] , x["charge"]
        x = torch.cat([self.emb(x0), self.emb(y0), self.emb(z0), self.emb(charge), self.emb(t0)], dim=-1)
        
        x = self.mlp(x) #here may be a linear layer, minor difference in performance observed
        return x
"""
196 + 1 cls token
width = 192 with 3 heads
layer = 8

the encoder block is implemented using torchscale library (default with sub-Layernorm)
"""
class base_model(nn.Module):
    def __init__(self, seq_length=196, hidden_dim=192):
        super().__init__()
        self.input_projection = InputProjection(seq_length=seq_length, hidden_features=hidden_dim*4, out_features=hidden_dim)
        
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02),requires_grad=True)
        self.class_token = nn.Parameter(torch.empty(1, 1, hidden_dim),requires_grad=True)

        encoder_config = EncoderConfig(
            encoder_attention_heads=3,
            encoder_embed_dim=hidden_dim,
            encoder_ffn_embed_dim=hidden_dim*4,
            encoder_layers=8,
            rel_pos_buckets=32,
            max_rel_pos=64
        )
        self.encoder = Encoder(encoder_config)
        self.proj = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        mask = x["mask"] 
        batch_size = mask.shape[0]
        
        x = self.input_projection(x)
        
        x += self.pos_embedding  #position embedding

        batch_class_token = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        x = self.encoder(src_tokens=None, token_embeddings=x)
        
        x = x['encoder_out']
        
        x = self.proj(x[:, 0, :])
        return x

"""
loss function, metric, optimizer
"""
def mse_loss(pred, y):
    pred_xyz = pred.float()
    
    sin_azi = torch.sin(y["target"][:, 0])
    cos_azi = torch.cos(y["target"][:, 0])
    sin_zen = torch.sin(y["target"][:, 1])
    cos_zen = torch.cos(y["target"][:, 1])
    true_xyz = torch.stack([cos_azi * sin_zen, sin_azi * sin_zen, cos_zen], -1)

    loss = nn.MSELoss()(pred_xyz, true_xyz)
    return loss

def mean_angular_error(pred, y):
    pred = F.normalize(pred.double(), dim=-1)

    sin_azi = torch.sin(y["target"][:, 0])
    cos_azi = torch.cos(y["target"][:, 0])
    sin_zen = torch.sin(y["target"][:, 1])
    cos_zen = torch.cos(y["target"][:, 1])

    cos_angle = (pred[:, 0] * cos_azi * sin_zen + pred[:, 1] * sin_azi * sin_zen + pred[:, 2] * cos_zen).clip(-1 , 1)
    loss = torch.acos(cos_angle).abs().mean(-1).float()
    return loss


def WrapperAdamW(param_groups,**kwargs):
    return OptimWrapper(param_groups,torch.optim.AdamW)
 
"""
for reproducibility purpose
"""                       
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train():
    seed_everything(37)
    
    train_set = Dataset(root_dir="ve_10_100TeV",mode='train')
    dl_train = torch.utils.data.DataLoader(train_set, batch_size=64, num_workers=8, pin_memory=True, pin_memory_device="cuda:0",persistent_workers=True)  # Set batch size to 32
    val_set = Dataset(root_dir="ve_10_100TeV",mode='val')
    dl_val = torch.utils.data.DataLoader(val_set, batch_size=64, num_workers=8)  # Set batch size to 32
    
    data = DataLoaders(dl_train, dl_val)
    
    model = base_model()
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters:", num_params)
    
    model = model.cuda()
    
    learn = Learner(
        data,
        model,
        path="trained_model",
        wd=0.01,
        loss_func=mse_loss,
        metrics=[mean_angular_error],
        opt_func=partial(WrapperAdamW, eps=1e-7),
        cbs=[GradientClip(3.0),
            CSVLogger(append=True),
            SaveModelCallback(every_epoch=False,with_opt=True), 
            GradientAccumulation(n_acc=2048//64)])  
    
    # load pretrained model if needed
    # learn.load("", with_opt=True)
    
    
    # learning rate increase from 1e-5 to 1e-3 then decrease to 1e-6
    # for more details refer to fastai api
    learn.fit_one_cycle(
            30,
            lr_max=1e-3,
            pct_start = 0.15,
            div = 100,
            div_final = 1000,        
        )

if __name__ == '__main__':
    train()
