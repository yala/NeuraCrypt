import pytorch_lightning as pl
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from pytorch_lightning import _logger as log
import sandstone.models.factory as model_factory
import sandstone.learn.metrics.factory as metric_factory
from sandstone.utils.generic import get_path_for_x, concat_all_gather
from sandstone.learn.losses.factory import get_loss
from sandstone.learn.lightning.factory import RegisterLightning, get_lightning
from sandstone.learn.lightning.default import Sandstone, prefix_dict
from collections import OrderedDict
from argparse import Namespace
import pickle
from sandstone.models.pools.factory import get_pool
import numpy as np
import copy
import remote_pdb
EPSILON = 1e-7

@RegisterLightning("private")
class SandstonePrivate(Sandstone):
    '''
    Lightning Module
    Methods:
        .log/.log_dict: log inputs to logger
    Notes:
        *_epoch_end method returns None
        self can log additional data structures to logger with self.logger.experiment.log_* (*= 'text', 'image', 'audio', 'confusion_matrix', 'histogram')
    '''
    def __init__(self, args):
        super(SandstonePrivate, self).__init__(args)

        if args.attack_snapshot is not None and args.attack_snapshot != "none":
            snapshot_args = Namespace(**pickle.load(open(args.attack_snapshot,'rb')))
            lightning_class = get_lightning(snapshot_args.lightning_name)
            model_w_hparams = lightning_class(snapshot_args)
            snapshot_args.skip_pos_embed_at_load = True
            attack_module = model_w_hparams.load_from_checkpoint(snapshot_args.model_path, args=snapshot_args)
            self.secure_encoder_0 = attack_module.attack_encoder
            self.secure_encoder_1 = attack_module.target_encoder
        else:
            self.secure_encoder_0 =  PrivateEncoder(args)
            self.secure_encoder_1 =  PrivateEncoder(args)

        for p in self.secure_encoder_0.parameters():
            p.requires_grad = False
        for p in self.secure_encoder_1.parameters():
            p.requires_grad = False

        self.exam_to_data = {}

    def step(self, batch, batch_idx, optimizer_idx, log_key_prefix = ""):
        batch['x'] = self.encode_input(batch['x'], batch['source'])
        return super().step(batch, batch_idx, optimizer_idx, log_key_prefix)

    @torch.no_grad()
    def encode_input(self, tensor, h_source = None):
        if h_source is None:
            return self.secure_encoder_0(tensor)

        B = h_source.size()[0]
        if 'transformer' in self.args.model_name :
            shape = [B, 1, 1]
        else:
            shape = [B, 1, 1, 1]
        h_source = h_source.view(shape)
        private_0 =  self.secure_encoder_0(tensor)
        private_1 =  self.secure_encoder_1(tensor)
        private = (h_source == 0) * private_0 + (h_source == 1) * private_1
        return private


class PrivateEncoder(nn.Module):
    def __init__(self, args, width_factor=1):
        super(PrivateEncoder, self).__init__()
        self.args = args
        input_dim = args.num_chan
        patch_size = args.private_kernel_size
        output_dim = args.hidden_dim
        num_patches =  (args.img_size[0] // patch_size) **2
        self.noise_size = 1

        args.input_dim = args.hidden_dim


        layers  = [
                    nn.Conv2d(input_dim, output_dim * width_factor, kernel_size=patch_size, dilation=1 ,stride=patch_size),
                    nn.ReLU()
                    ]
        for _ in range(self.args.private_depth):
            layers.extend( [
                nn.Conv2d(output_dim * width_factor, output_dim * width_factor , kernel_size=1, dilation=1, stride=1),
                nn.BatchNorm2d(output_dim * width_factor, track_running_stats=False),
                nn.ReLU()
            ])


        self.image_encoder = nn.Sequential(*layers)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, output_dim * width_factor))

        self.mixer = nn.Sequential( *[
            nn.ReLU(),
            nn.Linear(output_dim * width_factor, output_dim)
            ])


    def forward(self, x):
        encoded = self.image_encoder(x)
        B, C, H,W = encoded.size()
        encoded = encoded.view([B, -1, H*W]).transpose(1,2)
        encoded += self.pos_embedding
        encoded  = self.mixer(encoded)

        ## Shuffle indicies
        if not self.args.remove_pixel_shuffle:
            shuffled = torch.zeros_like(encoded)
            for i in range(B):
                idx = torch.randperm(H*W, device=encoded.device)
                for j, k in enumerate(idx):
                    shuffled[i,j] = encoded[i,k]
            encoded = shuffled

        return encoded
