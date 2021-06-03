import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
import pdb
import numpy as np
from sandstone.models.pools.factory import get_pool
from sandstone.models.factory import RegisterModel
from sandstone.models.dvit import DeepViT
from sandstone.learn.lightning.private import PrivateEncoder

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


@RegisterModel("reference_transformer")
class ReferenceTransformer(nn.Module):
    def __init__(self, args):

        super(ReferenceTransformer, self).__init__()
        self.args = args
        self.vit = DeepViT(image_size=args.img_size[0], patch_size=self.args.private_kernel_size, num_classes=args.num_classes, dim=args.hidden_dim,
            depth=args.num_layers, heads=args.num_heads, mlp_dim=args.hidden_dim * 2, dropout=args.dropout, emb_dropout=args.dropout)

        if not self.args.private:
            self.encoder = PrivateEncoder(args)

        if self.args.use_weak_encoder and not ( hasattr(args, 'skip_pos_embed_at_load') and args.skip_pos_embed_at_load ):
            num_patches =  (args.img_size[0] // args.private_kernel_size) **2
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, args.hidden_dim))

        if self.args.use_adv:
            self.class_emb = nn.Embedding(args.num_classes, args.hidden_dim)

    def forward(self, x, batch=None):
        """Computes a forward pass of the model.

        Arguments:
            x(Variable): The input to the model.

        Returns:
            The result of feeding the input through the model.
        """
        # Go through all layers up to fc
        if not self.args.private:
            x = self.encoder(x)

        if self.args.use_weak_encoder:
            x += self.pos_embedding

        if self.args.use_adv:
            ## Add token to represent class label
            side_info = self.class_emb( batch['class_label'].long()).unsqueeze(1)
            x = torch.cat( [x, side_info], dim=1)

        return {'logit': self.vit(x)}


