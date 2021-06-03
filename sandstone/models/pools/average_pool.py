import torch
import torch.nn as nn
from sandstone.models.pools.abstract_pool import AbstractPool
from sandstone.models.pools.factory import RegisterPool


@RegisterPool('GlobalAvgPool')
class GlobalAvgPool(AbstractPool):

    def replaces_fc(self):
        return False

    def forward(self, x, batch):
        spatially_flat_size = (*x.size()[:2], -1)
        x = x.view(spatially_flat_size)
        x = torch.mean(x, dim = -1)
        return None, x
