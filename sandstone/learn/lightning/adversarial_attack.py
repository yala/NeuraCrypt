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
from sandstone.learn.lightning.private import PrivateEncoder
from collections import OrderedDict
from argparse import Namespace
import pickle
from sandstone.models.pools.factory import get_pool
import numpy as np
import copy
import remote_pdb
EPSILON = 1e-7

@RegisterLightning("adversarial_attack")
class SandstoneAttack(Sandstone):
    '''
    Lightning Module
    Methods:
        .log/.log_dict: log inputs to logger
    Notes:
        *_epoch_end method returns None
        self can log additional data structures to logger with self.logger.experiment.log_* (*= 'text', 'image', 'audio', 'confusion_matrix', 'histogram')
    '''
    def __init__(self, args):
        super(SandstoneAttack, self).__init__(args)

        encoder_class = PrivateEncoder
        if args.use_weak_encoder:
            encoder_class = LinearEncoder

        self.attack_encoder =  encoder_class(args, width_factor=args.block_widening_factor)

        if args.attack_snapshot is not None:
            snapshot_args = Namespace(**pickle.load(open(args.attack_snapshot,'rb')))
            lightning_class = get_lightning(snapshot_args.lightning_name)
            model_w_hparams = lightning_class(snapshot_args)
            self.target_encoder = model_w_hparams.load_from_checkpoint(snapshot_args.model_path, args=snapshot_args).secure_encoder_0
        else:
            self.target_encoder = encoder_class(args)

        if self.args.attack_from_noise:
            self.noise_dist = torch.distributions.normal.Normal(0,1)

        for p in self.target_encoder.parameters():
            p.requires_grad = False


    def step(self, batch, batch_idx, optimizer_idx, log_key_prefix = ""):
        if self.args.use_same_dist:
            batch['source'] = torch.randint(high=2, size=batch['source'].size(), device=self.device)

        if self.args.use_shuffle_pairs:
            batch['x'], real, generated = self.encode_input_shuffle_pairs(batch['x'], batch['source'])
        else:
            batch['x'], real, generated = self.encode_input(batch['x'], batch['source'])

        batch['class_label'] = batch['y']

        model_output = self.model(batch['x'], batch=batch)
        logging_dict, predictions_dict = OrderedDict(), OrderedDict()


        if 'exam' in batch:
            predictions_dict['exam'] = batch['exam']

        predictions_dict['golds'] = batch['source']
        if optimizer_idx == 1:
            ## Generator step
            batch['y'] = torch.ones_like(batch['source'])
            loss_name = 'gen_loss'
        else:
            ## Discriminator step
            batch['y'] = batch['source']
            loss_name = 'disc_loss'
            logging_dict.update(self.log_attack_metrics(real, generated, batch['source']))

        loss_fn_name = 'cross_entropy'
        if self.args.use_mmd_adv:
            loss_fn_name = 'mmd_loss'
        if self.args.use_plaintext_attack:
            loss_fn_name = 'mse'
            model_output['logit'] = generated
            batch['y'] = real

        loss, _, local_predictions_dict = get_loss(loss_fn_name)(model_output, batch, self, self.args)
        if optimizer_idx != 1 and not self.args.use_plaintext_attack:
            predictions_dict.update(local_predictions_dict)
        logging_dict[loss_name] = loss

        logging_dict = prefix_dict(logging_dict, log_key_prefix)
        predictions_dict = prefix_dict(predictions_dict, log_key_prefix)

        return loss, logging_dict, predictions_dict, model_output

    def configure_optimizers(self):
        ## model is discriminator, attack is generator
        if self.args.use_mmd_adv or self.args.use_plaintext_attack:
            discrim_opt = []
        else:
            discrim_opt, _ = model_factory.get_optimizer(self.model, self.args)
        generator_opt, _ = model_factory.get_optimizer(self.attack_encoder, self.args)
        return discrim_opt + generator_opt, []

    def encode_input(self, tensor, h_source):
        B = h_source.size()[0]

        if 'transformer' in self.args.model_name :
            shape = [B, 1, 1]
        else:
            shape = [B, 1, 1, 1]
        h_source = h_source.view(shape)
        if self.args.attack_from_noise:
            noise_input = self.noise_dist.sample(tensor.size()).to(self.device)
            generated =  self.attack_encoder(noise_input)
        else:
            generated =  self.attack_encoder(tensor)
        with torch.no_grad():
            real =  self.target_encoder(tensor)
        private = (h_source == 1) * real + (h_source == 0) * generated
        return private, real, generated

    def encode_input_shuffle_pairs(self, tensor, h_source):
        B = h_source.size()[0]
        source, target = tensor[:,:,0], tensor[:,:,1]
        generated =  self.attack_encoder(source)
        with torch.no_grad():
            real =  self.target_encoder(target)

        if 'transformer' in self.args.model_name :
            shape = [B, 1, 1]
        else:
            shape = [B, 1, 1, 1]
        h_source = h_source.view(shape)
        private = (h_source == 1) * real + (h_source == 0) * generated
        return private, real, generated


    def log_attack_metrics(self, real, generated, source):
        logging_dict = {}

        is_public = source
        is_private = 1 - is_public

        def normalized_MSE(x, y):
            error = F.mse_loss(x, y, reduction='none').mean(dim=-1).mean(dim=-1)
            return error.mean().detach()

        logging_dict['overall_mse'] =  normalized_MSE(real, generated)

        real_mean = real.mean(dim=0).unsqueeze(0)
        logging_dict['pred_mean_baseline'] = normalized_MSE(real, real_mean)

        logging_dict['overall_mse/baseline'] = logging_dict['overall_mse'] / (logging_dict['pred_mean_baseline'] + 1e-9)

        gen_mean = generated.mean(dim=0).unsqueeze(0)
        logging_dict['mean_to_mean_mse'] = normalized_MSE(real_mean, gen_mean)

        shuffled_real = torch.zeros_like(real)
        perm = list(range(real.size()[0]))
        np.random.shuffle(perm)
        for i, j in enumerate(perm):
            shuffled_real[i] = real[j]

        logging_dict['permuation_mse'] = normalized_MSE(real, shuffled_real)
        return logging_dict


class LinearEncoder(nn.Module):
    def __init__(self, args, width_factor=1):
        super(LinearEncoder, self).__init__()
        self.args = args
        input_dim = args.num_chan
        patch_size = args.private_kernel_size
        output_dim = args.hidden_dim
        num_patches =  (args.img_size[0] // patch_size) **2
        self.noise_size = 1

        args.input_dim = args.hidden_dim


        layers  = [
                    nn.Conv2d(input_dim, output_dim, kernel_size=patch_size, dilation=1 ,stride=patch_size),
                    ]

        self.image_encoder = nn.Sequential(*layers)


    def forward(self, x):
        encoded = self.image_encoder(x)
        B, C, H,W = encoded.size()
        encoded = encoded.view([B, -1, H*W]).transpose(1,2)
        ## Shuffle indicies
        if not self.args.remove_pixel_shuffle:
            shuffled = torch.zeros_like(encoded)
            for i in range(B):
                idx = torch.randperm(H*W, device=encoded.device)
                for j, k in enumerate(idx):
                    shuffled[i,j] = encoded[i,k]
            encoded = shuffled
        return encoded
